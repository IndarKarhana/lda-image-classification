#!/usr/bin/env python3
"""
Generate publication-ready figures for the paper.

Figures 1-2 use the extended benchmark data (6 backbones x 3 datasets x 10 methods).
Figures 3-4 use Phase 3 data (component sweep, data efficiency -- CIFAR-100 only).
Figure 5 is a new heatmap for the fine-grained boundary condition.
"""

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Config
PROJECT = Path(__file__).resolve().parent.parent
FIGDIR = PROJECT / "paper" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
EXTENDED_DIR = PROJECT / "results" / "extended_benchmark"

# IEEE-style settings
plt.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Color palette
COLORS = {
    'Full':    '#7f7f7f',
    'PCA':     '#ff7f0e',
    'LDA':     '#1f77b4',
    'R-LDA':   '#9467bd',
    'LFDA':    '#d62728',
    'RDA':     '#2ca02c',
    'DSB':     '#e377c2',
    'NCA':     '#8c564b',
    'PCA+LDA': '#bcbd22',
    'RDA+SMD': '#17becf',
}

BACKBONE_ORDER = ['resnet18', 'resnet50', 'mobilenetv3', 'efficientnet', 'vit_b16', 'dinov2_vits14']
BACKBONE_LABELS = {
    'resnet18':       'R18',
    'resnet50':       'R50',
    'mobilenetv3':    'MV3',
    'efficientnet':   'EB0',
    'vit_b16':        'ViT',
    'dinov2_vits14':  'DiNO',
}
DATASET_ORDER = ['cifar100', 'tiny_imagenet', 'cub200']
DATASET_LABELS = {
    'cifar100':       'CIFAR-100',
    'tiny_imagenet':  'Tiny ImageNet',
    'cub200':         'CUB-200',
}


# ===== DATA LOADERS =====

def load_extended_benchmark():
    """Load all 18 LogReg CSV files from extended_benchmark/."""
    data = {}
    for csv_file in EXTENDED_DIR.glob("*_logreg.csv"):
        with open(csv_file) as f:
            for row in csv.DictReader(f):
                key = (row['backbone'], row['dataset'], row['method'])
                data[key] = {
                    'acc': float(row['accuracy_mean']),
                    'std': float(row['accuracy_std']),
                    'time_total': float(row['time_total']),
                    'time_reduce': float(row['time_reduce']),
                    'time_classify': float(row['time_classify']),
                    'dim': int(row['dim']),
                    'feature_dim': int(row['feature_dim']),
                    'category': row['category'],
                }
    return data


def load_component_sweep():
    """Load component sweep results from Phase 3."""
    rows = []
    path = PROJECT / "results" / "component_sweep" / "component_sweep.csv"
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    agg = defaultdict(list)
    for r in rows:
        key = (r['backbone'], r['method'], int(r['n_components']))
        agg[key].append(float(r['accuracy']))
    result = {}
    for key, accs in agg.items():
        result[key] = {'mean': np.mean(accs), 'std': np.std(accs, ddof=1) if len(accs) > 1 else 0}
    return result


def load_data_efficiency():
    """Load data efficiency results from Phase 3."""
    rows = []
    path = PROJECT / "results" / "phase3" / "data_efficiency.csv"
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    agg = defaultdict(list)
    for r in rows:
        key = (r['backbone'], r['dataset'], r['method'], float(r['fraction']))
        agg[key].append(float(r['accuracy']))
    result = {}
    for key, accs in agg.items():
        result[key] = {'mean': np.mean(accs), 'std': np.std(accs, ddof=1) if len(accs) > 1 else 0}
    return result


# ===== FIGURE 1: Accuracy Gain Over Full (3 panels) =====

def fig1_accuracy_gain():
    data = load_extended_benchmark()
    methods = ['PCA', 'LDA', 'DSB']
    method_colors = [COLORS[m] for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.6), sharey=True)

    for ax_idx, ds in enumerate(DATASET_ORDER):
        ax = axes[ax_idx]
        x = np.arange(len(BACKBONE_ORDER))
        width = 0.25

        for i, method in enumerate(methods):
            gains = []
            for bb in BACKBONE_ORDER:
                full_key = (bb, ds, 'Full')
                method_key = (bb, ds, method)
                if full_key in data and method_key in data:
                    gains.append(data[method_key]['acc'] - data[full_key]['acc'])
                else:
                    gains.append(0)
            ax.bar(x + (i - 1) * width, gains, width,
                   label=method, color=method_colors[i],
                   edgecolor='white', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([BACKBONE_LABELS[bb] for bb in BACKBONE_ORDER],
                           rotation=45, ha='right', fontsize=7)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_title(DATASET_LABELS[ds], fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        if ax_idx == 0:
            ax.set_ylabel('Accuracy gain over Full (%)')
            ax.legend(fontsize=7, loc='upper left', ncol=1, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(FIGDIR / "fig1_accuracy_gain.pdf")
    plt.savefig(FIGDIR / "fig1_accuracy_gain.png")
    plt.close()
    print("  -> Fig 1: Accuracy gain bar chart (3 panels x 6 backbones)")


# ===== FIGURE 2: Pareto Frontier (2 panels) =====

def fig2_pareto():
    data = load_extended_benchmark()
    methods_plot = ['Full', 'PCA', 'LDA', 'R-LDA', 'LFDA', 'RDA', 'DSB', 'NCA']

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))

    for ax_idx, ds in enumerate(['cifar100', 'tiny_imagenet']):
        ax = axes[ax_idx]
        method_points = {}

        for method in methods_plot:
            accs, times = [], []
            for bb in BACKBONE_ORDER:
                key = (bb, ds, method)
                if key in data:
                    accs.append(data[key]['acc'])
                    times.append(data[key]['time_total'])
            if accs:
                avg_acc = np.mean(accs)
                avg_time = np.mean(times)
                method_points[method] = (avg_time, avg_acc)
                marker = 'o' if method in ('Full', 'PCA', 'LDA') else 's'
                size = 60 if method in ('Full', 'PCA', 'LDA') else 40
                ax.scatter(avg_time, avg_acc, c=COLORS.get(method, '#333'),
                          s=size, marker=marker, label=method, zorder=5,
                          edgecolors='black', linewidth=0.5)

        # Pareto frontier
        pts = sorted(method_points.values(), key=lambda p: p[0])
        pareto = [pts[0]]
        for p in pts[1:]:
            if p[1] > pareto[-1][1]:
                pareto.append(p)
        if len(pareto) > 1:
            ax.plot([p[0] for p in pareto], [p[1] for p in pareto],
                    'k--', alpha=0.4, linewidth=1, zorder=1)

        ax.set_xlabel('Wall-clock time (s)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(DATASET_LABELS[ds])
        ax.grid(alpha=0.3)
        if ax_idx == 0:
            ax.legend(fontsize=6, loc='lower right', framealpha=0.9, ncol=2)

    plt.tight_layout()
    plt.savefig(FIGDIR / "fig2_pareto.pdf")
    plt.savefig(FIGDIR / "fig2_pareto.png")
    plt.close()
    print("  -> Fig 2: Pareto frontier (2 panels, 6 backbones)")


# ===== FIGURE 3: Component Sweep =====

def fig3_component_sweep():
    comp_data = load_component_sweep()
    ext_data = load_extended_benchmark()
    dims = [5, 10, 20, 40, 60, 80, 99]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))

    for ax_idx, (bb, bb_label) in enumerate([
        ('resnet18', 'ResNet-18 (512D)'),
        ('resnet50', 'ResNet-50 (2048D)')
    ]):
        ax = axes[ax_idx]

        for method, color, marker in [('LDA', COLORS['LDA'], 'o'), ('PCA', COLORS['PCA'], 's')]:
            means, stds = [], []
            for d in dims:
                key = (bb, method, d)
                if key in comp_data:
                    means.append(comp_data[key]['mean'])
                    stds.append(comp_data[key]['std'])
                else:
                    means.append(np.nan)
                    stds.append(0)
            means = np.array(means)
            stds = np.array(stds)
            ax.plot(dims, means, f'-{marker}', color=color, label=method,
                    markersize=5, linewidth=1.5)
            ax.fill_between(dims, means - stds, means + stds, alpha=0.15, color=color)

        # Full features baseline
        full_key = (bb, 'cifar100', 'Full')
        if full_key in ext_data:
            full_acc = ext_data[full_key]['acc']
            fdim = ext_data[full_key]['feature_dim']
            ax.axhline(y=full_acc, color=COLORS['Full'], linestyle='--',
                       linewidth=1, label=f'Full ({fdim}D)')

        ax.set_xlabel('Number of components ($d$)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(bb_label)
        ax.set_xticks(dims)
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGDIR / "fig3_component_sweep.pdf")
    plt.savefig(FIGDIR / "fig3_component_sweep.png")
    plt.close()
    print("  -> Fig 3: Component sweep")


# ===== FIGURE 4: Data Efficiency =====

def fig4_data_efficiency():
    data = load_data_efficiency()
    fractions = [0.1, 0.25, 0.5, 1.0]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))

    for ax_idx, (bb, bb_label) in enumerate([
        ('resnet18', 'ResNet-18 / CIFAR-100'),
        ('resnet50', 'ResNet-50 / CIFAR-100')
    ]):
        ax = axes[ax_idx]

        for method in ['Full', 'PCA', 'LDA', 'DSB']:
            means, stds = [], []
            for frac in fractions:
                key = (bb, 'cifar100', method, frac)
                if key in data:
                    means.append(data[key]['mean'])
                    stds.append(data[key]['std'])
                else:
                    means.append(np.nan)
                    stds.append(0)
            means = np.array(means)
            stds = np.array(stds)
            ax.plot([f * 100 for f in fractions], means, '-o', color=COLORS[method],
                    label=method, markersize=5, linewidth=1.5)
            ax.fill_between([f * 100 for f in fractions], means - stds, means + stds,
                            alpha=0.15, color=COLORS[method])

        ax.set_xlabel('Training data (%)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(bb_label)
        ax.set_xticks([10, 25, 50, 100])
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGDIR / "fig4_data_efficiency.pdf")
    plt.savefig(FIGDIR / "fig4_data_efficiency.png")
    plt.close()
    print("  -> Fig 4: Data efficiency")


# ===== FIGURE 5: Fine-Grained Boundary Condition Heatmap =====

def fig5_boundary_condition():
    data = load_extended_benchmark()

    gains = np.zeros((len(BACKBONE_ORDER), len(DATASET_ORDER)))
    for i, bb in enumerate(BACKBONE_ORDER):
        for j, ds in enumerate(DATASET_ORDER):
            full_key = (bb, ds, 'Full')
            lda_key = (bb, ds, 'LDA')
            if full_key in data and lda_key in data:
                gains[i, j] = data[lda_key]['acc'] - data[full_key]['acc']

    fig, ax = plt.subplots(figsize=(3.5, 3.2))

    vmax = max(abs(gains.min()), abs(gains.max()))
    im = ax.imshow(gains, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(DATASET_ORDER)))
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in DATASET_ORDER], fontsize=8)
    ax.set_yticks(range(len(BACKBONE_ORDER)))
    bb_full = ['ResNet-18', 'ResNet-50', 'MobileNetV3', 'EfficientNet', 'ViT-B/16', 'DINOv2']
    ax.set_yticklabels(bb_full, fontsize=8)

    for i in range(len(BACKBONE_ORDER)):
        for j in range(len(DATASET_ORDER)):
            val = gains[i, j]
            color = 'white' if abs(val) > vmax * 0.6 else 'black'
            sign = '+' if val > 0 else ''
            ax.text(j, i, f'{sign}{val:.1f}', ha='center', va='center',
                    fontsize=8, fontweight='bold', color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('LDA gain over Full (%)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title('LDA Accuracy Gain Over Full Features', fontsize=9, pad=8)

    plt.tight_layout()
    plt.savefig(FIGDIR / "fig5_boundary_condition.pdf")
    plt.savefig(FIGDIR / "fig5_boundary_condition.png")
    plt.close()
    print("  -> Fig 5: Boundary condition heatmap")


# ===== MAIN =====

if __name__ == "__main__":
    print("Generating paper figures...")
    print(f"  Source: {EXTENDED_DIR}")
    print(f"  Output: {FIGDIR}")
    print()
    fig1_accuracy_gain()
    fig2_pareto()
    fig3_component_sweep()
    fig4_data_efficiency()
    fig5_boundary_condition()
    print(f"\nAll figures saved to {FIGDIR}/")
    for f in sorted(FIGDIR.glob("fig*.pdf")):
        print(f"  {f.name}")
