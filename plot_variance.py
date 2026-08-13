"""Plot per-layer full-precision DPSR input variance on Set5."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from models import DPSR
from utils import create_val_loader


MODEL_DEFAULTS = {
    'scale': 2,
    'in_channels': 3,
    'num_blocks': 5,
    'full_channels': 32,
    'subnet_channels': 16,
}


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else Path(__file__).resolve().parent / path


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def latest_checkpoint(pattern):
    candidates = sorted((Path(__file__).resolve().parent / 'checkpoints').glob(pattern))
    return candidates[-1] if candidates else None


def build_full_precision_model(config, device):
    return DPSR(
        scale=config['scale'],
        in_dim=config['in_channels'],
        fea_dim=config['full_channels'],
        num_blocks=config['num_blocks'],
        bias=False,
        subnet_channels=config['subnet_channels'],
    ).to(device)


def checkpoint_config(saved, args):
    return {
        'scale': saved.get('scale', args.scale),
        'in_channels': saved.get('in_channels', args.in_channels),
        'num_blocks': saved.get('num_blocks', MODEL_DEFAULTS['num_blocks']),
        'full_channels': saved.get('full_channels', MODEL_DEFAULTS['full_channels']),
        'subnet_channels': saved.get('subnet_channels', MODEL_DEFAULTS['subnet_channels']),
    }




def load_model(args, device):
    weights_file = resolve_path(args.weights_file) if args.weights_file else latest_checkpoint(
        f'DPSR_x{args.scale}_*.pth'
    )
    if weights_file is None or not weights_file.is_file():
        raise FileNotFoundError('No full-precision DPSR checkpoint found; pass --weights-file explicitly.')
    checkpoint = load_checkpoint(weights_file, device)
    saved = checkpoint.get('model_config', {})
    if saved.get('quantizer', 'none') != 'none':
        raise ValueError(f'{weights_file} is not a full-precision DPSR checkpoint')
    config = checkpoint_config(saved, args)
    model = build_full_precision_model(config, device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    return model, config, weights_file


def computation_layer_names(model):
    return [
        name for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    ]

class FeatureMapSampler:
    """Sample feature-map inputs and track their exact streaming variance."""

    def __init__(self, model, names, max_samples, seed):
        self.max_samples = max_samples
        self.rng = np.random.default_rng(seed)
        self.values = {name: np.empty(0, dtype=np.float32) for name in names}
        self.counts = {name: 0 for name in names}
        self.means = {name: 0.0 for name in names}
        self.m2 = {name: 0.0 for name in names}
        modules = dict(model.named_modules())
        self.handles = [modules[name].register_forward_pre_hook(self._make_hook(name)) for name in names]

    def _make_hook(self, name):
        def hook(_module, inputs):
            if not inputs or not torch.is_tensor(inputs[0]):
                return
            values = inputs[0].detach().float().cpu().numpy().reshape(-1)
            self._add(name, values[np.isfinite(values)])
        return hook

    def _add(self, name, values):
        if not len(values):
            return
        old_count = self.counts[name]
        total_count = old_count + len(values)
        batch_mean = float(np.mean(values, dtype=np.float64))
        batch_m2 = float(np.var(values, dtype=np.float64) * len(values))
        delta = batch_mean - self.means[name]
        self.means[name] += delta * len(values) / total_count
        self.m2[name] += batch_m2 + delta ** 2 * old_count * len(values) / total_count
        target_size = min(self.max_samples, total_count)
        new_size = self.rng.hypergeometric(len(values), old_count, target_size)
        old_size = target_size - new_size
        old_indices = self.rng.choice(len(self.values[name]), old_size, replace=False) if old_size else []
        new_indices = self.rng.choice(len(values), new_size, replace=False) if new_size else []
        self.values[name] = np.concatenate((self.values[name][old_indices], values[new_indices]))
        self.counts[name] = total_count

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def summarize_layer(layer_name, values, input_count, mean, variance):
    return {
        'layer': layer_name,
        'samples': len(values),
        'input_values': input_count,
        'mean': mean,
        'variance': variance,
        'std': float(np.sqrt(variance)),
        'min': float(np.min(values)),
        'p01': float(np.percentile(values, 1)),
        'p05': float(np.percentile(values, 5)),
        'median': float(np.median(values)),
        'p95': float(np.percentile(values, 95)),
        'p99': float(np.percentile(values, 99)),
        'max': float(np.max(values)),
    }


def save_report(rows, output_path):
    with output_path.open('w', newline='', encoding='utf-8') as report_file:
        writer = csv.DictWriter(report_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_layer_variances(rows, output_path):
    variances = [row['variance'] for row in rows]
    layer_indices = np.arange(1, len(rows) + 1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(layer_indices, variances, color='#4C78A8')
    ax.set_xticks(layer_indices)
    ax.set_title('Full-precision DPSR per-layer input variance on Set5')
    ax.set_xlabel('Layer index')
    ax.set_ylabel('Input variance')
    ax.grid(axis='y', color='#D9D9D9', linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description='Plot full-precision DPSR per-layer input variance.')
    parser.add_argument('--weights-file', default='./checkpoints/DPSR_x2_0805_1549.pth', help='Full-precision DPSR checkpoint; defaults to the newest matching checkpoint.')
    parser.add_argument('--datasets-root', default='/home/tyzheng/Datasets_pt')
    parser.add_argument('--output-dir', default='distribution_plots/dpsr_input_variance')
    parser.add_argument('--scale', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--in-channels', type=int, default=3, choices=[1, 3])
    parser.add_argument('--top-k', type=int, default=0, help='Plot the first K layers only; 0 plots every layer.')
    parser.add_argument('--max-samples-per-layer', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.top_k < 0 or args.max_samples_per_layer < 1:
        raise ValueError('--top-k must be non-negative and --max-samples-per-layer positive')
    device = torch.device(args.device if args.device != 'cuda' or torch.cuda.is_available() else 'cpu')
    model, config, checkpoint_path = load_model(args, device)
    names = computation_layer_names(model)
    if args.top_k > len(names):
        raise ValueError(f'--top-k={args.top_k} exceeds available computation layers ({len(names)})')

    loader = create_val_loader(
        Path(args.datasets_root) / 'val' / 'Set5',
        config['scale'],
        in_channels=config['in_channels'],
    )

    sampler = FeatureMapSampler(model, names, args.max_samples_per_layer, args.seed)
    try:
        with torch.no_grad():
            for lr_img, _ in loader:
                model(lr_img.to(device).float() / 255.0)
    finally:
        sampler.close()

    report_rows = []
    for name in names:
        values = sampler.values[name]
        input_count = sampler.counts[name]
        variance = sampler.m2[name] / input_count
        stats = summarize_layer(name, values, input_count, sampler.means[name], variance)
        report_rows.append(stats)

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / 'layer_input_variance.csv'
    save_report(report_rows, report_path)
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Collected Set5 feature maps from {len(loader.dataset)} images.')
    print(f'All-layer report: {report_path}')
    selected = report_rows[:args.top_k] if args.top_k else report_rows
    output_path = output_dir / 'layer_input_variance.png'
    plot_layer_variances(selected, output_path)
    for layer_index, stats in enumerate(selected, start=1):
        print(f'Layer {layer_index:02d} | {stats["layer"]}: variance={stats["variance"]:.6g}')
    print(f'Variance plot: {output_path}')


if __name__ == '__main__':
    main()
