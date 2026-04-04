#!/usr/bin/env python3
"""Generate publication-ready figures for the paper."""

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
    'Full': '#7f7f7f',      # gray
    'PCA': '#ff7f0e',       # orange
    'LDA': '#1f77b4',       # blue
    'R-LDA': '#9467bd',     # purple
    'LFDA': '#d62728',      # red
    'RDA': '#2ca02c',       # green
    'DSB': '#e377c2',       # pink
    'NCA': '#8c564b',       # brown
    'PCA+LDA': '#bcbd22',   # olive
    'RDA+SMD': '#17becf',   # cyan
}


def load_multi_seed():
    """Load Phase 3 multi-seed results."""
    rows = []
    path = PROJECT / "results" / "phase3" / "multi_seed_results.csv"
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    
    agg = defaultdict(lambda: {'acc': [], 'time': []})
    for r in rows:
        key = (r['backbone'], r['dataset'], r['method'])
        agg[key]['acc'].append(float(r['accuracy']))
        agg[key]['time'].append(float(r['time_total']))
    
    result = {}
    for key, v in agg.items():
        result[key] = {
            'mean': np.mean(v['acc']),
            'std': np.std(v['acc'], ddof=1) if len(v['acc']) > 1 else 0,
            'time_mean': np.mean(v['time']),
        }
    return result


def load_component_sweep():
    """Load component sweep results."""
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
    """Load data efficiency results."""
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


def load_cost_analysis():
    """Load cost analysis results."""
    rows = []
    path = PROJECT / "results" / "phase3" / "cost_analysis.csv"
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


# ===== FIGURE 1: LDA Gain Over Full Features (Bar Chart) =====
def fig1_lda_gain():
    data = load_multi_seed()
    
    configs = [
        ('resnet18', 'cifar100', 'R18/C100'),
        ('resnet50', 'cifar100', 'R50/C100'),
        ('mobilenetv3', 'cifar100', 'MV3/C100'),
        ('efficientnet', 'cifar100', 'EB0/C100'),
        ('resnet18', 'tiny_imagenet', 'R18/TIN'),
        ('resnet50', 'tiny_imagenet', 'R50/TIN'),
        ('mobilenetv3', 'tiny_imagenet', 'MV3/TIN'),
        ('efficientnet', 'tiny_imagenet', 'EB0/TIN'),
    ]
    
    methods = ['PCA', 'LDA', 'DSB']
    x = np.arange(len(configs))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(7, 2.8))
    
    for i, method in enumerate(methods):
        gains = []
        for bb, ds, label in configs:
            full_acc = data[(bb, ds, 'Full')]['mean']
            method_acc = data[(bb, ds, method)]['mean']
            gains.append(method_acc - full_acc)
        
        bars = ax.bar(x + (i - 1) * width, gains, width, 
                      label=method, color=COLORS[method], 
                      edgecolor='white', linewidth=0.5)
    
    ax.set_ylabel('Accuracy gain over Full (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([c[2] for c in configs], rotation=30, ha='right')
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
    ax.legend(loc='upper right', ncol=3, framealpha=0.9)
    ax.set_title('Accuracy Improvement Over Full Features')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig1_accuracy_gain.pdf")
    plt.savefig(FIGDIR / "fig1_accuracy_gain.png")
    plt.close()
    print("  ✓ Fig 1: Accuracy gain bar chart")


# ===== FIGURE 2: Pareto Frontier =====
def fig2_pareto():
    data = load_multi_seed()
    methods = ['Full', 'PCA', 'LDA', 'R-LDA', 'LFDA', 'RDA', 'DSB']
    
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    
    for ax_idx, (dataset, ds_label) in enumerate([('cifar100', 'CIFAR-100'), ('tiny_imagenet', 'Tiny ImageNet')]):
        ax = axes[ax_idx]
        
        for method in methods:
            accs, times = [], []
            for bb in ['resnet18', 'resnet50', 'mobilenetv3', 'efficientnet']:
                key = (bb, dataset, method)
                if key in data:
                    accs.append(data[key]['mean'])
                    times.append(data[key]['time_mean'])
            
            avg_acc = np.mean(accs)
            avg_time = np.mean(times)
            
            marker = 'o' if method in ('Full', 'PCA', 'LDA') else 's'
            size = 60 if method in ('Full', 'PCA', 'LDA') else 40
            ax.scatter(avg_time, avg_acc, c=COLORS[method], s=size, 
                      marker=marker, label=method, zorder=5, edgecolors='black', linewidth=0.5)
        
        # Draw Pareto frontier for key methods
        points = []
        for method in methods:
            accs, times = [], []
            for bb in ['resnet18', 'resnet50', 'mobilenetv3', 'efficientnet']:
                key = (bb, dataset, method)
                if key in data:
                    accs.append(data[key]['mean'])
                    times.append(data[key]['time_mean'])
            points.append((np.mean(times), np.mean(accs), method))
        
        # Sort by time and find Pareto front
        points.sort(key=lambda p: p[0])
        pareto = [points[0]]
        for p in points[1:]:
            if p[1] > pareto[-1][1]:
                pareto.append(p)
        
        pareto_times = [p[0] for p in pareto]
        pareto_accs = [p[1] for p in pareto]
        ax.plot(pareto_times, pareto_accs, 'k--', alpha=0.4, linewidth=1, zorder=1)
        
        ax.set_xlabel('Wall-clock time (s)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(ds_label)
        ax.grid(alpha=0.3)
        if ax_idx == 0:
            ax.legend(fontsize=7, loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig2_pareto.pdf")
    plt.savefig(FIGDIR / "fig2_pareto.png")
    plt.close()
    print("  ✓ Fig 2: Pareto frontier")


# ===== FIGURE 3: Component Sweep =====
def fig3_component_sweep():
    data = load_component_sweep()
    dims = [5, 10, 20, 40, 60, 80, 99]
    
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    
    for ax_idx, (bb, bb_label, full_dim) in enumerate([
        ('resnet18', 'ResNet-18 (512D)', 512),
        ('resnet50', 'ResNet-50 (2048D)', 2048)
    ]):
        ax = axes[ax_idx]
        
        for method, color, marker in [('LDA', COLORS['LDA'], 'o'), ('PCA', COLORS['PCA'], 's')]:
            means = []
            stds = []
            for d in dims:
                key = (bb, method, d)
                if key in data:
                    means.append(data[key]['mean'])
                    stds.append(data[key]['std'])
                else:
                    means.append(np.nan)
                    stds.append(0)
            
            means = np.array(means)
            stds = np.array(stds)
            ax.plot(dims, means, f'-{marker}', color=color, label=method, markersize=5, linewidth=1.5)
            ax.fill_between(dims, means - stds, means + stds, alpha=0.15, color=color)
        
        # Full features baseline
        full_key = (bb, 'Full', full_dim)
        if full_key in data:
            full_acc = data[full_key]['mean']
            ax.axhline(y=full_acc, color=COLORS['Full'], linestyle='--', linewidth=1, label=f'Full ({full_dim}D)')
        
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
    print("  ✓ Fig 3: Component sweep")


# ===== FIGURE 4: Data Efficiency =====
def fig4_data_efficiency():
    data = load_data_efficiency()
    fractions = [0.1, 0.25, 0.5, 1.0]
    
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    
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
    print("  ✓ Fig 4: Data efficiency")


if __name__ == "__main__":
    print("Generating paper figures...")
    fig1_lda_gain()
    fig2_pareto()
    fig3_component_sweep()
    fig4_data_efficiency()
    print(f"\nAll figures saved to {FIGDIR}")
