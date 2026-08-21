import os

import torch
import torch.nn.functional as F

from models import DPSR, FSRCNN, channel_label
from utils import create_val_loader, final_test_parser, metrics
from utils.laplace import laplacian_map, rgb_to_gray


class DPSRFrame:
    def __init__(
        self,
        residual_model,
        scale,
        patch_size,
        overlap,
        full_threshold,
        subnet_threshold,
        device,
        subnet_channels=16,
    ):
        self.residual_model = residual_model
        self.scale = scale
        self.patch_size = patch_size
        self.overlap = overlap
        self.full_threshold = float(full_threshold)
        self.subnet_threshold = float(subnet_threshold)
        self.subnet_channels = int(subnet_channels)
        self.device = device
        self.stride = patch_size - overlap
        if self.stride <= 0:
            raise ValueError("arm_overlap 必须小于 arm_patch_size")
        if not self.subnet_threshold < self.full_threshold:
            raise ValueError("必须满足 arm_subnet_threshold < arm_threshold")

    def _starts(self, length):
        if length <= self.patch_size:
            return [0]
        starts = list(range(0, length - self.patch_size + 1, self.stride))
        tail = length - self.patch_size
        if starts[-1] != tail:
            starts.append(tail)
        return starts

    def _patch_smooth_score(self, laplace_scores, top, bottom, left, right):
        return laplace_scores[:, :, top:bottom, left:right].mean().item()

    def _residual(self, lr_patch, channels):
        return self.residual_model(lr_patch / 255.0, channels=channels) * 255.0

    def infer(self, lr_img):
        batch_size, channels, lr_h, lr_w = lr_img.shape
        if batch_size != 1 or channels != 3:
            raise ValueError(
                f"ARMSR expects a single RGB image, got shape {tuple(lr_img.shape)}"
            )

        accum = torch.zeros_like(
            F.interpolate(lr_img, scale_factor=self.scale, mode="nearest")
        )
        weight = torch.zeros_like(accum)
        total_patches = enhanced_patches = 0
        score_sum = 0.0
        branch_usage = {"bilinear": 0, "subnet": 0, "full": 0}

        with torch.no_grad():
            laplace_scores = laplacian_map(rgb_to_gray(lr_img))
            for top in self._starts(lr_h):
                for left in self._starts(lr_w):
                    bottom = min(top + self.patch_size, lr_h)
                    right = min(left + self.patch_size, lr_w)
                    actual_h, actual_w = bottom - top, right - left
                    lr_patch = lr_img[:, :, top:bottom, left:right]
                    if actual_h < self.patch_size or actual_w < self.patch_size:
                        lr_patch = F.pad(
                            lr_patch,
                            (
                                0,
                                self.patch_size - actual_w,
                                0,
                                self.patch_size - actual_h,
                            ),
                            mode="replicate",
                        )

                    score = self._patch_smooth_score(
                        laplace_scores, top, bottom, left, right
                    )
                    score_sum += score
                    total_patches += 1
                    base = (
                        F.interpolate(
                            lr_patch,
                            scale_factor=self.scale,
                            mode="bilinear",
                            align_corners=False,
                        )
                        .round()
                        .clamp(0, 255)
                    )
                    if score >= self.full_threshold:
                        branch_usage["full"] += 1
                        residual = self._residual(
                            lr_patch, channels=self.residual_model.fea_dim
                        )
                    elif score >= self.subnet_threshold:
                        branch_usage["subnet"] += 1
                        residual = self._residual(
                            lr_patch, channels=self.subnet_channels
                        )
                    else:
                        branch_usage["bilinear"] += 1
                        residual = None

                    if residual is None:
                        sr_patch = base
                    else:
                        sr_patch = (base + residual).round().clamp(0, 255)
                        enhanced_patches += 1
                    hr_top, hr_left = top * self.scale, left * self.scale
                    hr_h_valid, hr_w_valid = (
                        actual_h * self.scale,
                        actual_w * self.scale,
                    )
                    sr_valid = sr_patch[:, :, :hr_h_valid, :hr_w_valid]
                    accum[
                        :,
                        :,
                        hr_top : hr_top + hr_h_valid,
                        hr_left : hr_left + hr_w_valid,
                    ] += sr_valid
                    weight[
                        :,
                        :,
                        hr_top : hr_top + hr_h_valid,
                        hr_left : hr_left + hr_w_valid,
                    ] += 1.0

        sr_img = accum / weight.clamp_min(1.0)
        branch_ratios = {
            branch: count / max(1, total_patches)
            for branch, count in branch_usage.items()
        }
        return sr_img, {
            "total_patches": total_patches,
            "enhanced_patches": enhanced_patches,
            "enhanced_ratio": enhanced_patches / max(1, total_patches),
            "avg_laplace_score": score_sum / max(1, total_patches),
            "branch_usage": branch_usage,
            "branch_ratios": branch_ratios,
        }


def build_residual_model(args, device):
    model = DPSR(
        scale=args.scale,
        in_dim=3,
        fea_dim=args.channel_nums,
        num_blocks=args.num_blocks,
        bias=False,
        subnet_channels=args.subnet_channels,
    ).to(device)
    checkpoint = torch.load(args.dpsr_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    model.eval()
    return model


def build_fsrcnn_model(args, device):
    model = FSRCNN(scale_factor=args.scale, num_channels=3).to(device)
    checkpoint = torch.load(args.fsrcnn_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    model.eval()
    return model


def validate_fsrcnn_metrics(model, val_loader, scale, device, clip_ratio=1.0):
    metrics_list = []
    with torch.no_grad():
        for lr_img, hr_img in val_loader:
            lr_img = lr_img.to(device).float()
            hr_img = hr_img.to(device).float()
            sr_img = (model(lr_img / 255.0) * 255.0).round().clamp(0, 255)
            sr_img = sr_img[:, :, scale:-scale, scale:-scale]
            hr_img = hr_img[:, :, scale:-scale, scale:-scale]
            metrics_list.append(
                (
                    metrics.calculate_psnr(sr_img.squeeze(0), hr_img.squeeze(0)),
                    metrics.calculate_ssim(sr_img.squeeze(0), hr_img.squeeze(0)),
                )
            )
    if clip_ratio < 1.0:
        metrics_list.sort(key=lambda item: item[0], reverse=True)
        metrics_list = metrics_list[: max(1, int(len(metrics_list) * clip_ratio))]
    return {
        "psnr": sum(item[0] for item in metrics_list) / len(metrics_list),
        "ssim": sum(item[1] for item in metrics_list) / len(metrics_list),
    }


def build_fsrcnn_model(args, device):
    model = FSRCNN(scale_factor=args.scale, num_channels=3).to(device)
    checkpoint = torch.load(args.fsrcnn_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    model.eval()
    return model


def validate_fsrcnn_metrics(model, val_loader, scale, device, clip_ratio=1.0):
    metrics_list = []
    with torch.no_grad():
        for lr_img, hr_img in val_loader:
            lr_img = lr_img.to(device).float()
            hr_img = hr_img.to(device).float()
            sr_img = (model(lr_img / 255.0) * 255.0).round().clamp(0, 255)
            sr_img = sr_img[:, :, scale:-scale, scale:-scale]
            hr_img = hr_img[:, :, scale:-scale, scale:-scale]
            metrics_list.append((
                metrics.calculate_psnr(sr_img.squeeze(0), hr_img.squeeze(0)),
                metrics.calculate_ssim(sr_img.squeeze(0), hr_img.squeeze(0)),
            ))
    if clip_ratio < 1.0:
        metrics_list.sort(key=lambda item: item[0], reverse=True)
        metrics_list = metrics_list[: max(1, int(len(metrics_list) * clip_ratio))]
    return {
        "psnr": sum(item[0] for item in metrics_list) / len(metrics_list),
        "ssim": sum(item[1] for item in metrics_list) / len(metrics_list),
    }


def validate_arm_metrics(frame, val_loader, scale, clip_ratio=1.0):
    metrics_list = []
    total_patches = 0
    enhanced_patches = 0
    score_sum = 0.0
    branch_usage = {}

    with torch.no_grad():
        for lr_img, hr_img in val_loader:
            lr_img, hr_img = (
                lr_img.to(frame.device).float(),
                hr_img.to(frame.device).float(),
            )
            sr_img, stat = frame.infer(lr_img)
            crop_border = scale
            sr_img = sr_img[:, :, crop_border:-crop_border, crop_border:-crop_border]
            hr_img = hr_img[:, :, crop_border:-crop_border, crop_border:-crop_border]
            metrics_list.append(
                (
                    metrics.calculate_psnr(sr_img.squeeze(0), hr_img.squeeze(0)),
                    metrics.calculate_ssim(sr_img.squeeze(0), hr_img.squeeze(0)),
                )
            )
            total_patches += stat["total_patches"]
            enhanced_patches += stat["enhanced_patches"]
            score_sum += stat["avg_laplace_score"] * stat["total_patches"]
            for branch, count in stat["branch_usage"].items():
                branch_usage[branch] = branch_usage.get(branch, 0) + count

    if clip_ratio < 1.0:
        metrics_list.sort(key=lambda item: item[0], reverse=True)
        metrics_list = metrics_list[: max(1, int(len(metrics_list) * clip_ratio))]
    result = {
        "psnr": sum(item[0] for item in metrics_list) / len(metrics_list),
        "ssim": sum(item[1] for item in metrics_list) / len(metrics_list),
    }
    branch_ratios = {
        branch: count / max(1, total_patches) for branch, count in branch_usage.items()
    }
    return result, {
        "enhanced_ratio": enhanced_patches / max(1, total_patches),
        "avg_laplace_score": score_sum / max(1, total_patches),
        "enhanced_patches": enhanced_patches,
        "total_patches": total_patches,
        "branch_usage": branch_usage,
        "branch_ratios": branch_ratios,
    }


def main():
    args = final_test_parser()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    residual_model = build_residual_model(args, device)
    fsrcnn_model = build_fsrcnn_model(args, device)
    frame = DPSRFrame(
        residual_model=residual_model,
        scale=args.scale,
        patch_size=args.arm_patch_size,
        overlap=args.arm_overlap,
        full_threshold=args.arm_threshold,
        subnet_threshold=args.arm_subnet_threshold,
        device=device,
        subnet_channels=args.subnet_channels,
    )
    print(
        f"Routing: bilinear < {args.arm_subnet_threshold:g} <= "
        f"{channel_label(args.subnet_channels)} < {args.arm_threshold:g} <= "
        f"{channel_label(args.channel_nums)}"
    )
    val_loaders = {
        dataset_name: create_val_loader(
            os.path.join(args.val_root, dataset_name), args.scale, in_channels=3
        )
        for dataset_name in ("Set5", "Test4k")
    }
    for dataset_name, loader in val_loaders.items():
        dpsr_result, stat = validate_arm_metrics(
            frame, loader, args.scale, args.clip_ratio
        )
        fsrcnn_result = validate_fsrcnn_metrics(
            fsrcnn_model, loader, args.scale, device, args.clip_ratio
        )
        print(f"{dataset_name}:")
        print(
            f"  Dynamic DPSR: PSNR: {dpsr_result['psnr']:.2f}, "
            f"SSIM: {dpsr_result['ssim']:.4f}, "
            f"NNPatch: {stat['enhanced_ratio'] * 100:.2f}% "
            f"({stat['enhanced_patches']}/{stat['total_patches']}), "
            f"LapMean: {stat['avg_laplace_score']:.4f}"
        )
        usage_text = ", ".join(
            f"{branch}:{count}"
            for branch, count in sorted(stat["branch_usage"].items())
        )
        ratio_text = ", ".join(
            f"{branch}:{ratio * 100:.2f}%"
            for branch, ratio in sorted(stat["branch_ratios"].items())
        )
        print(f"  BranchUsage: {usage_text}")
        print(f"  BranchRatio: {ratio_text}")
        print(
            f"  FSRCNN: PSNR: {fsrcnn_result['psnr']:.2f}, "
            f"SSIM: {fsrcnn_result['ssim']:.4f}"
        )


if __name__ == "__main__":
    main()
