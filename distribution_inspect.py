"""
统计 DPSR / QDPSR 各层卷积输入分布与权重分布，可选保存直方图。
用法与 dis.sh 中参数一致。
"""

import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import DPSR
from models.QDPSR import QConv2dLSQP, QDPSR
from utils.dataloader import create_val_loader


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="DPSR", choices=["DPSR", "QDPSR"])
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--channel_nums", type=int, default=32)
    p.add_argument("--num_blocks", type=int, default=5)
    p.add_argument("--w_bits", type=int, default=4)
    p.add_argument("--a_bits", type=int, default=4)
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--val_dir", type=str, required=True)
    p.add_argument("--max_batches", type=int, default=20)
    p.add_argument(
        "--max_layer_plots",
        type=int,
        default=0,
        help=">0 时最多保存这么多层的输入直方图；<=0 表示每层都保存",
    )
    p.add_argument(
        "--max_weight_plots",
        type=int,
        default=0,
        help=">0 时最多保存这么多层的权重直方图；<=0 表示每层都保存",
    )
    p.add_argument(
        "--zoom_low_pct",
        type=float,
        default=1.0,
        help="zoom 图左侧分位数（如 1.0 表示 p1）",
    )
    p.add_argument(
        "--zoom_high_pct",
        type=float,
        default=99.0,
        help="zoom 图右侧分位数（如 99.0 表示 p99）",
    )
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _build_model(args, device):
    if args.model_name == "DPSR":
        return DPSR(
            scale=args.scale,
            in_dim=args.in_channels,
            fea_dim=args.channel_nums,
            num_blocks=args.num_blocks,
            bias=False,
        ).to(device)
    return QDPSR(
        scale=args.scale,
        in_dim=args.in_channels,
        fea_dim=args.channel_nums,
        num_blocks=args.num_blocks,
        bias=False,
        weight_bitwidth=args.w_bits,
        activation_bitwidth=args.a_bits,
    ).to(device)


def _iter_track_modules(model: nn.Module):
    """DPSR: nn.Conv2d；QDPSR: QConv2d（与旧脚本层名 body.0.filter1 一致，不用 .conv 后缀）。"""
    for name, m in model.named_modules():
        if isinstance(m, QConv2dLSQP):
            yield name, m
        elif isinstance(m, nn.Conv2d) and not name.endswith(".conv"):
            yield name, m


def _module_weight(mod):
    if isinstance(mod, QConv2dLSQP):
        return mod.conv.weight
    return mod.weight


def _percentiles(flat: np.ndarray):
    qs = [10, 25, 50, 75, 90]
    return [float(np.percentile(flat, q)) for q in qs]


def _save_hist(
    path: str,
    values: np.ndarray,
    title: str,
    bins: int = 80,
    color: str = "steelblue",
    log_y: bool = False,
):
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=bins, color=color, edgecolor="white", alpha=0.9)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    if log_y:
        plt.yscale("log")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _clip_by_percentile(values: np.ndarray, low_pct: float, high_pct: float):
    lo = float(np.percentile(values, low_pct))
    hi = float(np.percentile(values, high_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return values, float(values.min()), float(values.max())
    clipped = values[(values >= lo) & (values <= hi)]
    if clipped.size == 0:
        return values, float(values.min()), float(values.max())
    return clipped, lo, hi


def main():
    args = _parse_args()
    if not (0.0 <= args.zoom_low_pct < args.zoom_high_pct <= 100.0):
        raise ValueError("zoom 分位数参数必须满足 0 <= low < high <= 100")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = _build_model(args, device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    val_loader = create_val_loader(args.val_dir, args.scale, in_channels=args.in_channels)

    os.makedirs(args.save_dir, exist_ok=True)
    plot_dir = os.path.join(args.save_dir, "inputs_per_layer")
    weight_plot_dir = os.path.join(args.save_dir, "weights_per_layer")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(weight_plot_dir, exist_ok=True)

    input_acc = {}
    hooks = []

    def make_hook(layer_name):
        def hook(_mod, inp, _out):
            x = inp[0].detach()
            flat = x.reshape(-1).float().cpu().numpy()
            if layer_name not in input_acc:
                input_acc[layer_name] = []
            input_acc[layer_name].append(flat)

        return hook

    for name, mod in _iter_track_modules(model):
        hooks.append(mod.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for bi, (lr_img, _hr) in enumerate(val_loader):
            if bi >= args.max_batches:
                break
            lr_img = lr_img.to(device).float()
            model(lr_img / 255.0)

    for h in hooks:
        h.remove()

    input_csv_path = os.path.join(args.save_dir, "inputs_per_layer_stats.csv")
    weight_csv_path = os.path.join(args.save_dir, "weights_stats.csv")

    layer_names = sorted(input_acc.keys(), key=lambda s: (len(s), s))
    plot_budget = args.max_layer_plots
    weight_plot_budget = args.max_weight_plots
    plots_done = 0
    weight_plots_done = 0

    with open(input_csv_path, "w", newline="") as f_in, open(weight_csv_path, "w", newline="") as f_w:
        w_in = csv.writer(f_in)
        w_w = csv.writer(f_w)
        w_in.writerow(
            [
                "layer",
                "count",
                "min",
                "max",
                "mean",
                "std",
                "p10",
                "p25",
                "p50",
                "p75",
                "p90",
            ]
        )
        w_w.writerow(
            [
                "name",
                "count",
                "min",
                "max",
                "mean",
                "std",
                "p10",
                "p25",
                "p50",
                "p75",
                "p90",
            ]
        )

        for name in layer_names:
            parts = input_acc[name]
            if not parts:
                continue
            flat = np.concatenate(parts, axis=0)
            n = int(flat.shape[0])
            mn, mx = float(flat.min()), float(flat.max())
            mean, std = float(flat.mean()), float(flat.std())
            p10, p25, p50, p75, p90 = _percentiles(flat)
            w_in.writerow([name, n, mn, mx, mean, std, p10, p25, p50, p75, p90])

            do_plot = plot_budget <= 0 or plots_done < plot_budget
            if do_plot:
                safe_name = name.replace(".", "_").replace(os.path.sep, "_")
                png_name = safe_name + ".png"
                flat_zoom, _, _ = _clip_by_percentile(flat, args.zoom_low_pct, args.zoom_high_pct)
                _save_hist(
                    os.path.join(plot_dir, png_name),
                    flat_zoom,
                    title=f"Input Distribution (Adjusted Axis p{args.zoom_low_pct:.1f}-p{args.zoom_high_pct:.1f}): {name}",
                    bins=100,
                    log_y=True,
                )
                plots_done += 1

        weight_layer_names = []
        for name, mod in _iter_track_modules(model):
            w = _module_weight(mod).detach().float().cpu().numpy().reshape(-1)
            n = int(w.shape[0])
            mn, mx = float(w.min()), float(w.max())
            mean, std = float(w.mean()), float(w.std())
            p10, p25, p50, p75, p90 = _percentiles(w)
            w_w.writerow([f"{name}.weight", n, mn, mx, mean, std, p10, p25, p50, p75, p90])
            weight_layer_names.append((name, w))

        for name, w_arr in weight_layer_names:
            do_wplot = weight_plot_budget <= 0 or weight_plots_done < weight_plot_budget
            if do_wplot:
                safe_name = name.replace(".", "_").replace(os.path.sep, "_")
                png_name = safe_name + ".png"
                w_zoom, _, _ = _clip_by_percentile(w_arr, args.zoom_low_pct, args.zoom_high_pct)
                _save_hist(
                    os.path.join(weight_plot_dir, png_name),
                    w_zoom,
                    title=f"Weight Distribution (Adjusted Axis p{args.zoom_low_pct:.1f}-p{args.zoom_high_pct:.1f}): {name}",
                    bins=100,
                    color="coral",
                    log_y=True,
                )
                weight_plots_done += 1

    print(f"已写入: {input_csv_path}")
    print(f"已写入: {weight_csv_path}")
    print(f"输入直方图: {plot_dir} (共 {plots_done} 张)")
    print(f"权重直方图: {weight_plot_dir} (共 {weight_plots_done} 张)")


if __name__ == "__main__":
    main()
