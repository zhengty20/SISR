"""Collect actual INT4 layer-input quantization errors for QDPSR on Set5."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from models import QConv2dLSQP, build_qdpsr
from utils import create_val_loader

MODEL_DEFAULTS = {'scale': 2, 'in_channels': 3, 'num_blocks': 5, 'full_channels': 32, 'subnet_channels': 16, 'wbits': 4, 'abits': 4}


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else Path(__file__).resolve().parent / path


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def latest_checkpoint(pattern):
    checkpoints = sorted(resolve_path('checkpoints').glob(pattern))
    return checkpoints[-1] if checkpoints else None


def build_model(weights_file, args, device):
    checkpoint = load_checkpoint(weights_file, device)
    saved = checkpoint.get('model_config', {})
    if saved.get('quantizer') != 'lsqplus':
        raise ValueError(f'{weights_file} is not an LSQ+ QDPSR checkpoint')
    config = {key: saved.get(key, args.__dict__.get(key, value)) for key, value in MODEL_DEFAULTS.items()}
    model = build_qdpsr(scale=config['scale'], in_dim=config['in_channels'], fea_dim=config['full_channels'], num_blocks=config['num_blocks'], bias=False, weight_bitwidth=config['wbits'], activation_bitwidth=config['abits'], subnet_channels=config['subnet_channels']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    return model, config


def convolution_layer_names(model):
    return [name for name, module in model.named_modules() if isinstance(module, QConv2dLSQP)]


class ReservoirSampler:
    """Bounded uniform sample of a stream of scalar values."""
    def __init__(self, max_samples, rng):
        self.max_samples, self.rng = max_samples, rng
        self.values = np.empty(0, dtype=np.float32)
        self.count = 0
    def add(self, values):
        if not len(values):
            return
        total = self.count + len(values)
        target = min(self.max_samples, total)
        new_count = self.rng.hypergeometric(len(values), self.count, target)
        old_count = target - new_count
        retained = self.values[self.rng.choice(len(self.values), old_count, replace=False)] if old_count else np.empty(0, dtype=np.float32)
        added = values[self.rng.choice(len(values), new_count, replace=False)].astype(np.float32, copy=False) if new_count else np.empty(0, dtype=np.float32)
        self.values = np.concatenate((retained, added))
        self.count = total


class LayerInputQuantErrorSampler:
    """Collect Q(x) - x at each convolution's actual INT4 network input."""
    def __init__(self, model, layer_names, max_samples, seed):
        modules = dict(model.named_modules())
        rng = np.random.default_rng(seed)
        self.samplers = {name: ReservoirSampler(max_samples, rng) for name in layer_names}
        self.handles = [modules[name].register_forward_pre_hook(self._make_hook(name, modules[name])) for name in layer_names]
    def _make_hook(self, name, module):
        def hook(_module, inputs):
            if not inputs or not torch.is_tensor(inputs[0]):
                return
            x = inputs[0].detach()
            error = (module._maybe_quantize_input(x) - x).float().cpu().numpy().reshape(-1)
            self.samplers[name].add(error[np.isfinite(error)])
        return hook
    def close(self):
        for handle in self.handles:
            handle.remove()


def save_samples(layer_names, samplers, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='', encoding='utf-8') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=('layer_index', 'layer', 'sample_index', 'noise'))
        writer.writeheader()
        for index, name in enumerate(layer_names, start=1):
            writer.writerows({'layer_index': index, 'layer': name, 'sample_index': sample_index, 'noise': float(value)} for sample_index, value in enumerate(samplers[name].values, start=1))


def plot_boxplot(layer_names, samplers, output_path):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    boxplot = ax.boxplot([samplers[name].values for name in layer_names], positions=np.arange(1, len(layer_names) + 1), widths=.58, patch_artist=True, showfliers=False)
    for box in boxplot['boxes']:
        box.set(facecolor='#A6C8E0', edgecolor='#4C78A8')
    ax.axhline(0, color='#222222')
    ax.set_xlabel('Convolution Layer Index')
    ax.set_ylabel('Layer-input Quantization Error')
    ax.grid(axis='y', color='#D9D9D9')
    ax.set_axisbelow(True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def print_statistics(layer_names, samplers):
    for index, name in enumerate(layer_names, start=1):
        sampler = samplers[name]
        noise = sampler.values
        print(f'Layer {index:02d} | {name}: samples={sampler.count}, mean={noise.mean():.6g}, std={noise.std():.6g}, median={np.median(noise):.6g}, p01={np.percentile(noise, 1):.6g}, p99={np.percentile(noise, 99):.6g}')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights-file', default='')
    parser.add_argument('--datasets-root', default='/home/tyzheng/Datasets_pt')
    parser.add_argument('--output-file', default='distribution_plots/qdpsr_convolution_noise/convolution_input_quant_error_boxplot.png')
    parser.add_argument('--output-csv', default='distribution_plots/qdpsr_convolution_noise/convolution_input_quant_error_samples.csv')
    parser.add_argument('--scale', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--in-channels', type=int, default=3, choices=[1, 3])
    parser.add_argument('--max-samples-per-layer', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_samples_per_layer < 1:
        raise ValueError('--max-samples-per-layer must be positive')
    weights_file = resolve_path(args.weights_file) if args.weights_file else latest_checkpoint(f'QDPSR_x{args.scale}_*.pth')
    if weights_file is None or not weights_file.is_file():
        raise FileNotFoundError('No QDPSR checkpoint found; pass --weights-file explicitly.')
    device = torch.device(args.device if args.device != 'cuda' or torch.cuda.is_available() else 'cpu')
    model, config = build_model(weights_file, args, device)
    layer_names = convolution_layer_names(model)
    collector = LayerInputQuantErrorSampler(model, layer_names, args.max_samples_per_layer, args.seed)
    loader = create_val_loader(Path(args.datasets_root) / 'val' / 'Set5', config['scale'], in_channels=config['in_channels'])
    try:
        with torch.no_grad():
            for lr_image, _ in loader:
                model(lr_image.to(device).float() / 255.0)
    finally:
        collector.close()
    output_path, csv_path = resolve_path(args.output_file), resolve_path(args.output_csv)
    plot_boxplot(layer_names, collector.samplers, output_path)
    save_samples(layer_names, collector.samplers, csv_path)
    print_statistics(layer_names, collector.samplers)
    print(f'Noise samples: {csv_path}')


if __name__ == '__main__':
    main()
