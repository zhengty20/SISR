from pathlib import Path

import torch
import torch.nn as nn

from utils import create_val_loader, test_parser, validate_metrics, bicubic_metrics
from models import build_qdpsr

WEIGHT_BITWIDTH = 4
ACTIVATION_BITWIDTH = 4
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
        return module.__class__.__name__ == 'QConv2dLSQP' or isinstance(module, (nn.PReLU, nn.PixelShuffle))

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


def plot_distribution(name, tensor, output_dir=OUTPUT_DIR):
    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    except ImportError as exc:
        print(f'Skipped distribution plot for {name}: {exc.name} is not installed.')
        return

    arr = tensor.numpy().astype(np.float32)
    values = arr.reshape(-1)

    full_x_min, full_x_max = robust_xlim(values)

    # Main plot focuses on the readable high-density region.
    # This avoids wasting 70% of the plot on a nearly empty long tail.
    q_low, q_high = np.percentile(values, [1.0, 99.7])
    x_min = max(q_low, -3.0)
    x_max = min(q_high, 2.0)

    clipped = values[(values >= x_min) & (values <= x_max)]

    fig, ax = plt.subplots(figsize=(3.2, 2.15))

    counts, edges = np.histogram(
        clipped,
        bins=min(BINS, 100),
        range=(x_min, x_max),
        density=True,
    )
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Light smoothing makes the distribution readable after Visio scaling.
    if len(counts) >= 5:
        kernel = np.ones(5, dtype=np.float32) / 5
        counts_smooth = np.convolve(counts, kernel, mode='same')
    else:
        counts_smooth = counts

    color = '#2f7fb8'
    ax.fill_between(centers, counts_smooth, color=color, alpha=0.35, linewidth=0)
    ax.plot(centers, counts_smooth, color=color, linewidth=1.8)

    ax.axvline(0, color='black', linewidth=0.9, alpha=0.75)
    ax.axvspan(-1, 1, color='#2ca02c', alpha=0.07, linewidth=0)

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel('Activation value', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)

    ax.tick_params(axis='both', labelsize=8, width=0.8, length=3)
    ax.grid(visible=True, axis='y', color='gray', linestyle='--', linewidth=0.35, alpha=0.45)
    ax.grid(visible=False, axis='x')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.9)
    ax.spines['bottom'].set_linewidth(0.9)

    ax.text(
        0.04, 0.92,
        'Dense near 0',
        transform=ax.transAxes,
        fontsize=8.5,
        weight='bold',
        va='top',
    )

    # Inset: preserve the full long-tail message without ruining main readability.
    axins = inset_axes(ax, width='38%', height='35%', loc='upper left', borderpad=0.9)

    full_clipped = values[(values >= full_x_min) & (values <= full_x_max)]
    axins.hist(
        full_clipped,
        bins=min(BINS, 120),
        density=True,
        color=color,
        alpha=0.45,
        linewidth=0,
    )
    axins.set_xlim(full_x_min, full_x_max)
    axins.set_xticks([round(full_x_min), 0])
    axins.set_yticks([])
    axins.tick_params(axis='x', labelsize=6, width=0.6, length=2)
    axins.spines['top'].set_visible(False)
    axins.spines['right'].set_visible(False)
    axins.spines['left'].set_visible(False)
    axins.spines['bottom'].set_linewidth(0.6)
    axins.set_title('full range', fontsize=6.5, pad=1)

    ax.annotate(
        'long tail',
        xy=(x_min + 0.08 * (x_max - x_min), max(counts_smooth) * 0.10),
        xytext=(x_min + 0.22 * (x_max - x_min), max(counts_smooth) * 0.42),
        arrowprops=dict(arrowstyle='->', linewidth=0.8),
        fontsize=7.5,
    )

    plt.tight_layout(pad=0.45)

    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = safe_layer_name(name)
    output_path = output_dir / f'{base_name}_distribution_compact.png'
    svg_path = output_dir / f'{base_name}_distribution_compact.svg'

    plt.savefig(output_path, dpi=300)
    plt.savefig(svg_path)
    plt.close()

    print(
        f'Saved {output_path} and {svg_path} | '
        f'shape={arr.shape}, '
        f'min={values.min():.6f}, max={values.max():.6f}, '
        f'mean={values.mean():.6f}, std={np.std(values):.6f}, '
        f'main_xlim=({x_min:.6f}, {x_max:.6f}), '
        f'full_xlim=({full_x_min:.6f}, {full_x_max:.6f})'
    )


def plot_layer_inputs(layer_inputs):
    for name, tensor in layer_inputs.items():
        plot_distribution(name, tensor)


def _build_val_loaders(scale, in_channels):
    return {
        'Set5': create_val_loader('/home/tyzheng/Datasets_pt/val/Set5', scale, in_channels=in_channels),
        # 'Set14': create_val_loader('/home/tyzheng/Datasets_pt/val/Set14', scale, in_channels=in_channels),
        # 'B100': create_val_loader('/home/tyzheng/Datasets_pt/val/B100', scale, in_channels=in_channels),
        # 'U100': create_val_loader('/home/tyzheng/Datasets_pt/val/U100', scale, in_channels=in_channels),
        # 'M109': create_val_loader('/home/tyzheng/Datasets_pt/val/M109', scale, in_channels=in_channels),
    }


def _print_metrics(loaders, metric_fn, name_suffix=''):
    for dataset_name, loader in loaders.items():
        result = metric_fn(loader)
        label = f'{dataset_name}{name_suffix}'
        print(f'{label}: PSNR: {result["psnr"]:.2f}, SSIM: {result["ssim"]:.4f}')


if __name__ == '__main__':

    args = test_parser()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    net = build_qdpsr(
        scale=args.scale,
        in_dim=args.in_channels,
        fea_dim=args.channel_nums,
        num_blocks=args.num_blocks,
        bias=False,
        weight_bitwidth=WEIGHT_BITWIDTH,
        activation_bitwidth=ACTIVATION_BITWIDTH,
    ).to(device)

    net.head.set_input_quantization(False)
    checkpoint = torch.load("./checkpoints/QDPSR_x2_0521_2101.pth", map_location=device, weights_only=False)
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
    net.load_state_dict(model_state_dict)
    net.eval()
    collector = InputDistributionCollector(net, dump_layers=DUMP_LAYERS)
    print(f'Distribution plots will be saved to: {OUTPUT_DIR}')

    val_loaders = _build_val_loaders(args.scale, args.in_channels)

    _print_metrics(
        loaders=val_loaders,
        metric_fn=lambda loader: validate_metrics(net, loader, args.scale, device, 1.0),
    )
    collector.close()
    collector.print()
    plot_layer_inputs(collector.plot_inputs)

    _print_metrics(
        loaders=val_loaders,
        metric_fn=lambda loader: bicubic_metrics(loader, args.scale, device),
        name_suffix='-bicubic',
    )
