# Supervised Dimensionality Reduction Revisited: Why LDA on Frozen CNN Features Deserves a Second Look

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Paper: IEEE TNNLS](https://img.shields.io/badge/Paper-IEEE%20TNNLS-green.svg)](paper/)

> **Indar Kumar, Girish Karhana, Sai Krishna Jasti, Ankit Hemant Lade**
>
> We investigate whether reducing the dimensionality of frozen CNN features before classification improves downstream accuracy. Through controlled experiments spanning 4 backbones, 2 datasets, and 10 methods, we find that classical LDA consistently **improves** accuracy over full features by up to 4.6 percentage points while reducing dimensionality by 61–95%.

---

## Key Findings

| Finding | Evidence |
|---------|----------|
| **LDA beats full features in 8/8 configs** | +0.26% to +4.58%, all p < 0.001 |
| **LDA beats PCA in 7/8 configs** | Supervised signal provides +1.9 to +3.9% lift |
| **LDA beats LFDA, NCA in 8/8 configs** | Simpler method wins; 3–25× faster |
| **Full features are never Pareto-optimal** | Slower *and* less accurate than LDA |
| **DSB (ours) wins 4/8 configs** | +0.2–0.4% over LDA at 2–3× cost |
| **Results generalize to Tiny ImageNet** | Consistent across 100- and 200-class tasks |

---

## Methods Compared (10 total)

| Category | Method | Description |
|----------|--------|-------------|
| **Control** | Full | No reduction (baseline) |
| | PCA | Unsupervised variance-based projection |
| **Classical** | LDA | Fisher's supervised projection (d = C−1) |
| | PCA+LDA | Two-stage: PCA then LDA |
| **Academic** | R-LDA | Regularized LDA with Ledoit–Wolf shrinkage |
| | LFDA | Local Fisher Discriminant Analysis |
| | NCA | Neighbourhood Components Analysis |
| **Ours** | RDA | Residual Discriminant Augmentation |
| | DSB | Discriminant Subspace Boosting |
| | RDA+SMD | RDA with Spectral Margin Discriminants |

---

## Experimental Setup

### Backbones (frozen, ImageNet-pretrained)

| Backbone | Feature Dim | Role |
|----------|------------|------|
| ResNet-18 | 512 | Compact residual baseline |
| ResNet-50 | 2048 | High-dimensional test |
| MobileNetV3-Small | 576 | Edge deployment |
| EfficientNet-B0 | 1280 | Modern efficient |

### Datasets

| Dataset | Classes | Train | Test | Max LDA Dim (C−1) |
|---------|---------|-------|------|--------------------|
| CIFAR-100 | 100 | 50,000 | 10,000 | 99 |
| Tiny ImageNet | 200 | 100,000 | 10,000 | 199 |

### Classifier

L2-regularized logistic regression (LBFGS, max_iter=5000, C=1.0) with StandardScaler applied after projection. Identical hyperparameters across all methods for fair comparison.

---

## Results (5-seed means ± std)

### CIFAR-100

| Method | ResNet-18 (512D) | ResNet-50 (2048D) | MobileNetV3 (576D) | EfficientNet (1280D) |
|--------|-----------------|-------------------|--------------------|-----------------------|
| Full | 62.85±.02 | 71.98±.00 | 65.58±.07 | 71.41±.04 |
| PCA | 65.06±.01 | 69.18±.12 | 64.61±.01 | 69.77±.13 |
| **LDA** | **66.97±.00** | 72.24±.05 | 68.47±.03 | 72.30±.04 |
| DSB | **67.27±.00** | 72.62±.06 | **68.88±.04** | **72.55±.00** |

### Tiny ImageNet

| Method | ResNet-18 (512D) | ResNet-50 (2048D) | MobileNetV3 (576D) | EfficientNet (1280D) |
|--------|-----------------|-------------------|--------------------|-----------------------|
| Full | 59.84±.04 | 74.29±.00 | 59.24±.06 | 71.79±.00 |
| PCA | 64.46±.02 | 73.35±.08 | 61.95±.04 | 71.99±.17 |
| **LDA** | 64.42±.06 | 75.01±.01 | 63.35±.01 | 72.34±.06 |
| DSB | **64.70±.00** | 75.24±.06 | **63.73±.00** | 72.14±.06 |

Full 10-method tables with timing are in the paper (Tables I and II).

---

## Project Structure

```
lda-image-classification/
├── data/
│   ├── load_cifar100.py              # CIFAR-100 loader
│   └── tiny_imagenet.py              # Tiny ImageNet Dataset class
├── features/
│   ├── extract_features.py           # ResNet-18 feature extraction
│   ├── extract_features_multi.py     # Multi-backbone extraction (CIFAR-100)
│   ├── extract_tiny_imagenet.py      # Multi-backbone extraction (Tiny ImageNet)
│   └── saved/                        # Cached features (.npz, gitignored)
├── reduction/
│   ├── lda.py                        # LDA wrapper
│   ├── pca.py                        # PCA wrapper
│   ├── regularized_lda.py            # R-LDA with Ledoit-Wolf shrinkage
│   ├── cw_lda.py                     # Confusion-Weighted LDA
│   ├── dg_lda.py                     # DG-LDA orchestrator
│   ├── feature_profiler.py           # Feature space geometry analysis
│   └── adaptive_components.py        # Automatic component selection
├── models/
│   └── linear_classifier.py          # LogisticRegression wrapper
├── experiments/
│   ├── run_academic_benchmark.py     # Main: 10 methods × 4 backbones × 2 datasets
│   ├── run_smoke_test.py             # Quick validation
│   ├── run_novel_methods_test.py     # RDA/DSB/SMD development
│   ├── run_tiny_imagenet.py          # Tiny ImageNet generalization
│   ├── run_dglda_experiment.py       # DG-LDA framework experiments
│   └── run_feature_profiling.py      # Feature space profiling
├── scripts/
│   ├── generate_paper_figures.py     # Publication figure generation
│   ├── run_all_benchmarks.sh         # Full benchmark runner
│   └── extract_tiny_gcp.py           # GCP feature extraction helper
├── paper/                            # IEEE TNNLS manuscript (LaTeX)
│   ├── main.tex
│   ├── references.bib
│   ├── sections/                     # 7 section files
│   └── figures/                      # 4 publication figures (PDF + PNG)
├── results/
│   ├── academic_benchmark/           # Phase 2: 80 single-seed results
│   ├── phase3/                       # Multi-seed, significance, data efficiency, cost
│   ├── component_sweep/              # LDA/PCA dimension sweep
│   ├── novel_methods/                # RDA/DSB development results
│   └── smoke_test/                   # Validation results
├── notebooks/
│   └── comprehensive_analysis.ipynb  # Visualization & analysis
├── archive/                          # Legacy experiments from initial study
├── requirements.txt
├── LICENSE                           # MIT
└── README.md
```

---

## Reproduction

### Setup

```bash
git clone https://github.com/IndarKarhana/lda-image-classification.git
cd lda-image-classification
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run Benchmark

```bash
# Extract features (first run only; ~5 min per backbone on MPS/CUDA)
python features/extract_features_multi.py

# Full 10-method benchmark (single backbone)
python experiments/run_academic_benchmark.py --backbone resnet18 --dataset cifar100

# Generate paper figures
python scripts/generate_paper_figures.py
```

### Compile Paper

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

---

## Statistical Rigor

- **5 random seeds** for multi-seed methods; single-seed methods marked with † in tables
- **Paired t-tests** and **Wilcoxon signed-rank tests** for all LDA vs. method comparisons
- **Pareto analysis** for accuracy–cost tradeoff (LDA: 7/8 Pareto-optimal; Full: 0/8)
- **Data efficiency study**: LDA crossover at ~25–50% training data; below that, full features preferred
- **Component sweep**: d = C−1 confirmed optimal; monotonic improvement, no overfitting

---

## Citation

```bibtex
@article{kumar2026lda,
  title={Supervised Dimensionality Reduction Revisited: Why {LDA} on Frozen {CNN}
         Features Deserves a Second Look},
  author={Kumar, Indar and Karhana, Girish and Jasti, Sai Krishna and Lade, Ankit Hemant},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2026},
  note={Under review}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
