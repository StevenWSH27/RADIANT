"""Generate the README performance dashboard from paper table values."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASETS = ["Lytro", "MFI-WHU", "DSCIE", "Scene", "LLVIP", "TNO", "QuickBird", "Harvard"]
GIF_EI = np.array([80.86, 91.61, 111.27, 101.32, 62.46, 52.30, 23.02, 100.71])
RADIANT_EI = np.array([85.55, 97.93, 118.73, 108.20, 66.08, 55.33, 24.85, 107.06])


def main() -> None:
    assets = Path(__file__).resolve().parents[1] / "assets"
    output_svg = assets / "performance_summary.svg"
    output_png = assets / "performance_summary.png"
    gains = 100.0 * (RADIANT_EI - GIF_EI) / GIF_EI

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    figure = plt.figure(figsize=(13.2, 7.2), facecolor="#081525")
    grid = figure.add_gridspec(2, 4, height_ratios=[1, 3.1], hspace=0.38, wspace=0.18)

    cards = [
        ("3.108 MB", "model size", "#58d8ff"),
        ("9.42", "GFLOPs @ 256²", "#a994ff"),
        ("42 FPS", "single-image inference", "#ff82cc"),
        ("96.8%", "FLIR mAP@0.5", "#65e6a5"),
    ]
    for index, (value, label, color) in enumerate(cards):
        axis = figure.add_subplot(grid[0, index])
        axis.set_facecolor("#10243a")
        axis.text(
            0.5,
            0.63,
            value,
            ha="center",
            va="center",
            color=color,
            fontsize=24,
            fontweight="bold",
        )
        axis.text(
            0.5,
            0.28,
            label,
            ha="center",
            va="center",
            color="#c9d8eb",
            fontsize=10,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color(color)
            spine.set_alpha(0.5)

    axis = figure.add_subplot(grid[1, :])
    axis.set_facecolor("#0d2034")
    positions = np.arange(len(DATASETS))
    colors = plt.cm.cool(np.linspace(0.15, 0.85, len(DATASETS)))
    bars = axis.barh(positions, gains, color=colors, height=0.62)
    axis.set_yticks(positions)
    axis.set_yticklabels(DATASETS, color="#edf5ff")
    axis.invert_yaxis()
    axis.set_xlabel("Relative EI gain over GIF (%)", color="#c9d8eb", labelpad=10)
    axis.set_title(
        "Consistent gains across 8 benchmarks / 6 fusion regimes",
        color="white",
        fontsize=17,
        fontweight="bold",
        pad=16,
    )
    axis.grid(axis="x", color="white", alpha=0.12, linestyle="--")
    axis.tick_params(axis="x", colors="#c9d8eb")
    for name in ("top", "right", "left"):
        axis.spines[name].set_visible(False)
    axis.spines["bottom"].set_color("#46627f")
    axis.set_xlim(0, max(gains) + 1.2)
    for bar, gain in zip(bars, gains):
        axis.text(
            bar.get_width() + 0.12,
            bar.get_y() + bar.get_height() / 2,
            f"+{gain:.1f}%",
            va="center",
            color="white",
            fontweight="bold",
        )

    figure.text(
        0.5,
        0.015,
        "Values reproduced from the paper's main quantitative table; "
        "GIF is the strongest consistently available universal baseline.",
        ha="center",
        color="#8fa8c1",
        fontsize=9,
    )
    figure.savefig(
        output_svg, format="svg", facecolor=figure.get_facecolor(), bbox_inches="tight"
    )
    figure.savefig(
        output_png,
        format="png",
        dpi=170,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
    )
    print(output_svg)
    print(output_png)


if __name__ == "__main__":
    main()
