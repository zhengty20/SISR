import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = SCRIPT_DIR / "spatial_redundancy_plots" / "laplace_psnr_ad.csv"
REQUIRED_COLUMNS = (
    "laplacian_center",
    "dpsr_psnr_db",
    "baseline_psnr_db",
    "bicubic_psnr_db",
)
OPTIONAL_COLUMNS = (
    "dpsr_compressed_psnr_db",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Laplacian-binned PSNR values from a CSV file."
    )
    parser.add_argument("csv_path", type=Path, nargs="?", default=DEFAULT_CSV_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for SVG files (defaults to the CSV directory).",
    )
    return parser.parse_args()


def load_statistics(csv_path):
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header row.")
        missing_columns = set(REQUIRED_COLUMNS) - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                f"{csv_path} is missing required columns: {', '.join(sorted(missing_columns))}"
            )
        rows = list(reader)
        columns = list(REQUIRED_COLUMNS)
        columns.extend(
            column for column in OPTIONAL_COLUMNS if column in reader.fieldnames
        )

    if not rows:
        raise ValueError(f"{csv_path} has no data rows.")

    return {
        column: np.asarray([float(row[column]) for row in rows])
        for column in columns
    }


def save_line_plot(stats, output_path, include_bicubic):
    matplotlib.use("Agg")
    plt.rcParams["font.family"] = "Arial"
    centers = stats["laplacian_center"]
    dpsr = stats["dpsr_psnr_db"]
    dpsr_compressed = stats.get("dpsr_compressed_psnr_db")
    baseline = stats["baseline_psnr_db"]
    bicubic = stats["bicubic_psnr_db"]

    valid = np.isfinite(centers) & np.isfinite(dpsr) & np.isfinite(baseline)
    if dpsr_compressed is not None:
        valid &= np.isfinite(dpsr_compressed)
    if include_bicubic:
        valid &= np.isfinite(bicubic)
    if valid.sum() < 2:
        raise ValueError("At least two complete PSNR rows are required.")

    x = centers[valid]
    fig, ax = plt.subplots(figsize=(4.9, 4.2), constrained_layout=True)
    ax.plot(x, baseline[valid], color="#AD65DC", linewidth=3, label="Baseline")
    ax.plot(x, dpsr[valid], color="#8ddb51", linewidth=3, label="DPSR (full)")
    if dpsr_compressed is not None:
        ax.plot(
            x,
            dpsr_compressed[valid],
            color="#f09a45",
            linewidth=3,
            linestyle="--",
            label="DPSR (0.5x, x=3)",
        )
    if include_bicubic:
        ax.plot(x, bicubic[valid], color="#5a89e6", linewidth=3, label="Bicubic")
        ax.set_xlim(0, 13.5)
        ax.set_ylim(32, 54)
    else:
        # ax.plot(x, bicubic[valid], color="#5a89e6", linewidth=3, label="Bicubic")
        ax.set_xlim(6.5, 13.5)
        ax.set_ylim(34, 40)
        # ax.axvline(7, color='red', linewidth=2.2, alpha=0.65)      

    ax.set_xlabel("Laplacian Magnitude", fontsize=22, labelpad=4, fontweight="bold")
    ax.set_ylabel("PSNR (dB)", fontsize=22, labelpad=4, fontweight="bold")
    ax.tick_params(axis="both", labelsize=20)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(color="gray", linewidth=1.0, alpha=0.65, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1, 1),
        borderaxespad=0,
        frameon=False,
        prop={"size": 16, "weight": "bold"},
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight", format="svg")
    plt.close(fig)


def main():
    args = parse_args()
    if not args.csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")

    output_dir = args.output_dir or args.csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = load_statistics(args.csv_path)
    if "dpsr_compressed_psnr_db" not in stats:
        print("No compressed DPSR column found; plotting the full DPSR curve only.")

    full_plot_path = output_dir / "laplace_psnr1.svg"
    comparison_plot_path = output_dir / "laplace_psnr2.svg"
    save_line_plot(stats, full_plot_path, include_bicubic=True)
    save_line_plot(stats, comparison_plot_path, include_bicubic=False)
    print(f"Read {len(stats['laplacian_center'])} PSNR intervals from {args.csv_path}")
    print(f"Saved {full_plot_path}")
    print(f"Saved {comparison_plot_path}")


if __name__ == "__main__":
    main()
