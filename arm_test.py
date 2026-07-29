import os
import torch
import torch.nn.functional as F

from models import DPSR, QDPSR
from utils import create_val_loader, metrics
from utils.arm_test_parser import arm_test_parser


class ARMSRFrame:
    def __init__(
        self,
        residual_model,
        scale,
        patch_size,
        overlap,
        threshold,
        device,
    ):
        self.residual_model = residual_model
        self.scale = scale
        self.patch_size = patch_size
        self.overlap = overlap
        self.threshold = threshold
        self.device = device
        self.stride = patch_size - overlap
        if self.stride <= 0:
            raise ValueError("arm_overlap 必须小于 arm_patch_size")
        lap = torch.tensor([[0.0, 0.25, 0.0], [0.25, -1.0, 0.25], [0.0, 0.25, 0.0]], device=device)
        self.laplace_kernel = lap.view(1, 1, 3, 3)


    def _starts(self, length):
        if length <= self.patch_size:
            return [0]
        starts = list(range(0, length - self.patch_size + 1, self.stride))
        tail = length - self.patch_size
        if starts[-1] != tail:
            starts.append(tail)
        return starts

    def _patch_smooth_score(self, patch):
        y = 0.299 * patch[:, 0:1] + 0.587 * patch[:, 1:2] + 0.114 * patch[:, 2:3]
        lap = F.conv2d(y, self.laplace_kernel, padding=1)
        return lap.abs().mean().item()

    def infer(self, lr_img):
        batch_size, channels, lr_h, lr_w = lr_img.shape
        if batch_size != 1 or channels != 3:
            raise ValueError(f'ARMSR expects a single RGB image, got shape {tuple(lr_img.shape)}')

        hr_h, hr_w = lr_h * self.scale, lr_w * self.scale
        accum = torch.zeros_like(F.interpolate(lr_img, scale_factor=self.scale, mode='nearest'))
        weight = torch.zeros_like(accum)

        total_patches = 0
        enhanced_patches = 0
        score_sum = 0.0
        branch_usage = {"bilinear": 0, "model": 0}

        h_starts = self._starts(lr_h)
        w_starts = self._starts(lr_w)

        with torch.no_grad():
            for top in h_starts:
                for left in w_starts:
                    bottom = min(top + self.patch_size, lr_h)
                    right = min(left + self.patch_size, lr_w)
                    actual_h = bottom - top
                    actual_w = right - left
                    lr_patch = lr_img[:, :, top:bottom, left:right]
                    if actual_h < self.patch_size or actual_w < self.patch_size:
                        pad_bottom = self.patch_size - actual_h
                        pad_right = self.patch_size - actual_w
                        lr_patch = F.pad(lr_patch, (0, pad_right, 0, pad_bottom), mode='replicate')

                    score = self._patch_smooth_score(lr_patch)
                    score_sum += score
                    total_patches += 1

                    base = F.interpolate(lr_patch, scale_factor=self.scale, mode='bilinear', align_corners=False).round().clamp(0, 255)
                    if score >= self.threshold:
                        branch_usage["model"] += 1
                        residual = self.residual_model(lr_patch / 255.0) * 255.0
                        sr_patch = (base + residual).round().clamp(0, 255)
                        enhanced_patches += 1
                    else:
                        branch_usage["bilinear"] += 1
                        sr_patch = base.clamp(0, 255)

                    hr_top = top * self.scale
                    hr_left = left * self.scale
                    hr_h_valid = actual_h * self.scale
                    hr_w_valid = actual_w * self.scale
                    sr_valid = sr_patch[:, :, :hr_h_valid, :hr_w_valid]
                    accum[:, :, hr_top:hr_top + hr_h_valid, hr_left:hr_left + hr_w_valid] += sr_valid
                    weight[:, :, hr_top:hr_top + hr_h_valid, hr_left:hr_left + hr_w_valid] += 1.0

        sr_img = accum / weight.clamp_min(1.0)
        stats = {
            "total_patches": total_patches,
            "enhanced_patches": enhanced_patches,
            "enhanced_ratio": enhanced_patches / max(1, total_patches),
            "avg_laplace_score": score_sum / max(1, total_patches),
            "branch_usage": branch_usage,
        }
        return sr_img, stats


def build_residual_model(args, device):  
    model = DPSR(
        scale=args.scale,
        in_dim=3,
        fea_dim=args.channel_nums,
        num_blocks=args.num_blocks,
        bias=False,
    ).to(device)
    checkpoint_path = "./checkpoints/DPSR_x2_0729_1531.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(model_state_dict)
    model.eval()
    return model


def validate_arm_metrics(frame, val_loader, scale, device, clip_ratio=1.0):
    metrics_list = []
    total_patches = 0
    enhanced_patches = 0
    score_sum = 0.0
    branch_usage = {}

    with torch.no_grad():
        for lr_img, hr_img in val_loader:
            lr_img, hr_img = lr_img.to(device).float(), hr_img.to(device).float()
            sr_img, stat = frame.infer(lr_img)

            crop_border = scale
            sr_img = sr_img[:, :, crop_border:-crop_border, crop_border:-crop_border]
            hr_img = hr_img[:, :, crop_border:-crop_border, crop_border:-crop_border]

            psnr = metrics.calculate_psnr(sr_img.squeeze(0), hr_img.squeeze(0))
            ssim = metrics.calculate_ssim(sr_img.squeeze(0), hr_img.squeeze(0))
            metrics_list.append((psnr, ssim))

            total_patches += stat["total_patches"]
            enhanced_patches += stat["enhanced_patches"]
            score_sum += stat["avg_laplace_score"] * stat["total_patches"]
            for branch, cnt in stat.get("branch_usage", {}).items():
                branch_usage[branch] = branch_usage.get(branch, 0) + cnt

    if clip_ratio < 1.0:
        metrics_list.sort(key=lambda x: x[0], reverse=True)
        selected_count = max(1, int(len(metrics_list) * clip_ratio))
        selected_metrics = metrics_list[:selected_count]
    else:
        selected_metrics = metrics_list

    psnr_list = [item[0] for item in selected_metrics]
    ssim_list = [item[1] for item in selected_metrics]
    result = {
        "psnr": sum(psnr_list) / len(psnr_list),
        "ssim": sum(ssim_list) / len(ssim_list),
    }
    stats = {
        "enhanced_ratio": enhanced_patches / max(1, total_patches),
        "avg_laplace_score": score_sum / max(1, total_patches),
        "enhanced_patches": enhanced_patches,
        "total_patches": total_patches,
        "branch_usage": branch_usage,
    }
    return result, stats


def main():
    args = arm_test_parser()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    residual_model = build_residual_model(args, device)
    frame = ARMSRFrame(
        residual_model=residual_model,
        scale=args.scale,
        patch_size=args.arm_patch_size,
        overlap=args.arm_overlap,
        threshold=args.arm_threshold,
        device=device,
    )

    val_loaders = {
        "Set5": create_val_loader(os.path.join(args.val_root, "Set5"), args.scale, in_channels=3)
    }

    for dataset_name, loader in val_loaders.items():
        result, stat = validate_arm_metrics(frame, loader, args.scale, device, args.clip_ratio)
        print(
            f'{dataset_name}: PSNR: {result["psnr"]:.2f}, SSIM: {result["ssim"]:.4f}, '
            f'EnhPatch: {stat["enhanced_ratio"] * 100:.2f}% ({stat["enhanced_patches"]}/{stat["total_patches"]}), '
            f'LapMean: {stat["avg_laplace_score"]:.4f}'
        )
        usage_text = ", ".join([f"{branch}:{cnt}" for branch, cnt in sorted(stat["branch_usage"].items())])
        print(f"BranchUsage: {usage_text}")

if __name__ == "__main__":
    main()