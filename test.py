from pathlib import Path

import torch
import torch.nn as nn

from utils import create_val_loader, test_parser, validate_metrics, bicubic_metrics
from models import DPSR

DUMP_LAYERS = ('body.1.projection2', 'body.3.filter2')
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / 'distribution_plots'
BINS = 160
LOW_Q = 0.001
HIGH_Q = 0.999


class InputDistributionCollector:
    def __init__(self, model, max_forwards=3, dump_layers=None):
        self.handles = []
        self.inputs = {}
        self.dump_layers = set(dump_layers or [])
        self.plot_inputs = {}
        self.max_forwards = max_forwards
        self.forward_count = 0
        self.collected = False

        for name, module in model.named_modules():
            if name and self._should_collect(module):
                self.inputs[name] = []
                self.handles.append(module.register_forward_pre_hook(self._make_hook(name)))
        unmatched_layers = self.dump_layers - set(self.inputs)
        if unmatched_layers:
            print(f'Warning: dump layers not found or not collectable: {sorted(unmatched_layers)}')
        self.handles.append(model.register_forward_hook(self._count_forward))

    @staticmethod
    def _should_collect(module):
        return isinstance(module, (nn.Conv2d, nn.PReLU, nn.PixelShuffle))

    def _make_hook(self, name):
        def hook(module, inputs):
            if self.collected:
                return
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return

            first_image = x[:1].detach().float().cpu()
            self.inputs[name].append(first_image.flatten())
            if name in self.dump_layers and name not in self.plot_inputs:
                self.plot_inputs[name] = first_image.clone()
        return hook

    def _count_forward(self, module, inputs, output):
        self.forward_count += 1
        if self.forward_count >= self.max_forwards:
            self.collected = True

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def print(self):
        print('\nFirst image layer input distributions:')
        for name, values in self.inputs.items():
            if not values:
                print(f'{name}: no tensor input collected')
                continue
            x = torch.cat(values)
            q01, q50, q99 = torch.quantile(x, torch.tensor([0.01, 0.50, 0.99]))
            print(
                f'{name}: '
                f'numel={x.numel()}, '
                f'mean={x.mean().item():.6f}, '
                f'std={x.std(unbiased=False).item():.6f}, '
                f'min={x.min().item():.6f}, '
                f'p01={q01.item():.6f}, '
                f'p50={q50.item():.6f}, '
                f'p99={q99.item():.6f}, '
                f'max={x.max().item():.6f}'
            )


def safe_layer_name(name):
    return name.replace('.', '_')


def robust_xlim(values):
    import numpy as np

    lo, hi = np.quantile(values, [LOW_Q, HIGH_Q])
    if np.isclose(lo, hi):
        lo, hi = values.min(), values.max()
    if np.isclose(lo, hi):
        pad = max(abs(float(lo)) * 0.1, 1e-3)
        return float(lo - pad), float(hi + pad)

    pad = float(hi - lo) * 0.08
    return float(lo - pad), float(hi + pad)


def smooth_hist(values, bins=160, xlim=None, sigma_bins=1.2):
    import numpy as np

    if xlim is None:
        xlim = (values.min(), values.max())

    counts, edges = np.histogram(values, bins=bins, range=xlim, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    radius = max(1, int(3 * sigma_bins))
    kx = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (kx / sigma_bins) ** 2)
    kernel /= kernel.sum()
    counts = np.convolve(counts, kernel, mode='same')

    return centers, counts


def plot_distribution(name, tensor, core_xlim=(-0.3, 0.3), output_dir=OUTPUT_DIR):
    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f'Skipped distribution plot for {name}: {exc.name} is not installed.')
        return

    plt.rcParams["font.family"] = "Arial"
    values = tensor.numpy().astype(np.float32).reshape(-1)
    x_min, x_max = robust_xlim(values)

    full = values[(values >= x_min) & (values <= x_max)]
    core = values[(values >= core_xlim[0]) & (values <= core_xlim[1])]

    if core.size == 0:
        core = values
        core_xlim = (float(values.min()), float(values.max()))

    fig, ax = plt.subplots(figsize=(4.2, 3.45))

    color = "#3c973b"
    xs, ys = smooth_hist(core, bins=140, xlim=core_xlim)
    ax.fill_between(xs, ys, color=color, alpha=0.35, linewidth=0)
    ax.plot(xs, ys, color=color, linewidth=3)

    ax.axvline(0, color='black', linewidth=1.0, alpha=0.65)

    ax.set_xlim(*core_xlim)
    ax.set_xlabel('Activation value', fontsize=22, labelpad=4, fontweight="bold")
    ax.set_ylabel('Density', fontsize=22, labelpad=4, fontweight="bold")
    ax.tick_params(labelsize=20)

    ax.grid(visible=True, axis='y', color='gray', linestyle='--', linewidth=1, alpha=0.65)
    ax.grid(visible=False, axis='x')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_layer_name(name)
    png_path = output_dir / f'{stem}_distribution_compact.png'

    fig.tight_layout(pad=0.4)
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(
        f'Saved {png_path}| '
        f'shape={values.shape}, '
        f'min={values.min():.6f}, max={values.max():.6f}, '
        f'mean={values.mean():.6f}, std={np.std(values):.6f}, '
        f'full_xlim=({x_min:.6f}, {x_max:.6f}), '
        f'core_xlim=({core_xlim[0]:.2f}, {core_xlim[1]:.2f})'
    )


def plot_layer_inputs(layer_inputs):
    for name, tensor in layer_inputs.items():
        plot_distribution(name, tensor)


def _build_val_loaders(scale, in_channels, val_root, datasets):
    return {
        dataset_name: create_val_loader(
            str(Path(val_root) / dataset_name), scale, in_channels=in_channels
        )
        for dataset_name in datasets
    }


def _print_width_metrics(model, loaders, args, device):
    for dataset_name, loader in loaders.items():
        for active_num_blocks in args.block_nums:
            width_results = {}
            for width_mult in args.width_mults:
                result = validate_metrics(
                    model,
                    loader,
                    args.scale,
                    device,
                    1.0,
                    is_residual=True,
                    width_mult=width_mult,
                    active_num_blocks=active_num_blocks,
                )
                width_results[width_mult] = result
                if width_mult == 1.0:
                    width_description = f'channels={args.channel_nums}'
                else:
                    width_description = (
                        f'head_channels={args.channel_nums // 2}, '
                        f'expand_block={args.subnet_expand_block}'
                    )
                print(
                    f'{dataset_name} | blocks={active_num_blocks} | '
                    f'width={width_mult:.1f} | {width_description}: '
                    f'PSNR={result["psnr"]:.4f} dB, SSIM={result["ssim"]:.6f}'
                )

            if 1.0 in width_results:
                full_result = width_results[1.0]
                for width_mult, result in width_results.items():
                    if width_mult == 1.0:
                        continue
                    print(
                        f'{dataset_name} | blocks={active_num_blocks} | '
                        f'width={width_mult:.1f} vs full: '
                        f'PSNR_delta={result["psnr"] - full_result["psnr"]:+.4f} dB, '
                        f'SSIM_delta={result["ssim"] - full_result["ssim"]:+.6f}'
                    )


if __name__ == '__main__':

    args = test_parser()
    args.block_nums = tuple(sorted(set(int(depth) for depth in args.block_nums)))
    if not args.block_nums or any(depth < 1 or depth > args.num_blocks for depth in args.block_nums):
        raise ValueError(f'--block_nums must be in [1, {args.num_blocks}], got {args.block_nums}')
    if 0.5 in args.width_mults and args.subnet_expand_block > min(args.block_nums):
        raise ValueError('--subnet_expand_block must not exceed the smallest tested depth')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    net = DPSR(
        scale=args.scale,
        in_dim=args.in_channels,
        fea_dim=args.channel_nums,
        num_blocks=args.num_blocks,
        bias=False,
        subnet_expand_block=args.subnet_expand_block,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
    net.load_state_dict(model_state_dict, strict=True)
    net.eval()
    collector = InputDistributionCollector(net, dump_layers=DUMP_LAYERS) if args.collect_distributions else None
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Device: {device}')
    if collector is not None:
        print(f'Distribution plots will be saved to: {OUTPUT_DIR}')

    val_loaders = _build_val_loaders(
        args.scale,
        args.in_channels,
        args.val_root,
        args.datasets,
    )

    _print_width_metrics(net, val_loaders, args, device)
    if collector is not None:
        collector.close()
        collector.print()
        plot_layer_inputs(collector.plot_inputs)
