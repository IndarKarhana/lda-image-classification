# Supervised Dimensionality Reduction Revisited: Why LDA on Frozen CNN Features Deserves a Second Look

[![arXiv](https://img.shields.io/badge/arXiv-2604.03928-b31b1b.svg)](https://arxiv.org/abs/2604.03928)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)

📄 **Paper:** [arXiv:2604.03928](https://arxiv.org/abs/2604.03928)

> **Indar Kumar, Girish Karhana, Sai Krishna Jasti, Ankit Hemant Lade**
>
> We investigate whether reducing the dimensionality of frozen pretrained features before classification improves downstream accuracy. Through 180 controlled experiments spanning **6 backbones** (4 CNNs + 2 vision transformers), **3 datasets** (CIFAR-100, Tiny ImageNet, CUB-200), and **10 methods**, we find that classical LDA consistently **improves** accuracy on coarse-grained tasks by up to 4.5 percentage points — but **loses** on fine-grained CUB-200, establishing a clear boundary condition.

---

## Key Findings

| Finding | Evidence |
|---------|----------|
| **LDA beats full features on 11/12 coarse-grained configs** | +0.08% to +4.46%, all p < 0.001 |
| **Fine-grained boundary condition** | LDA loses on all 6 CUB-200 configs (−2.1% to −7.1%) |
| **LDA beats PCA in 10/12 coarse-grained configs** | Supervised signal provides +0.3 to +3.9% lift |
| **Full features are never Pareto-optimal (coarse)** | Slower *and* less accurate than LDA |
| **DSB (ours) wins 6/18 configs outright** | +0.2–0.5% over LDA at 2–3× cost |
| **DINOv2 at 384D beats all other backbones** | 87.73% on CUB-200 despite lowest dimensionality |
| **8.4× average speedup** | CIFAR-100 avg across 6 backbones |

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

| Backbone | Feature Dim | Type |
|----------|------------|------|
| ResNet-18 | 512 | CNN |
| ResNet-50 | 2048 | CNN |
| MobileNetV3-Small | 576 | CNN |
| EfficientNet-B0 | 1280 | CNN |
| ViT-B/16 | 768 | Transformer (supervised) |
| DINOv2 ViT-S/14 | 384 | Transformer (self-supervised) |

### Datasets

| Dataset | Classes | Train | Test | Task Type |
|---------|---------|-------|------|-----------|
| CIFAR-100 | 100 | 50,000 | 10,000 | Coarse-grained |
| Tiny ImageNet | 200 | 100,000 | 10,000 | Coarse-grained |
| CUB-200-2011 | 200 | 5,994 | 5,794 | Fine-grained (birds) |

### Classifier

L2-regularized logistic regression (LBFGS, max_iter=5000, C=1.0) with StandardScaler applied after projection. Identical hyperparameters across all methods for fair comparison.

---

## Results (Selected — 5-seed means)

### CIFAR-100 (LDA helps ✅)

| Method | R18 (512D) | R50 (2048D) | MV3 (576D) | EB0 (1280D) | ViT (768D) | DiNO (384D) |
|--------|-----------|------------|-----------|------------|-----------|------------|
| Full | 62.85 | 72.06 | 65.69 | 71.58 | 78.81 | 80.72 |
| PCA | 65.07 | 69.14 | 64.65 | 69.77 | 78.05 | 81.55 |
| **LDA** | **66.97** | 72.29 | 68.51 | 72.30 | 78.79 | 82.37 |
| DSB | 67.20 | 72.52 | **68.94** | **72.63** | 79.10 | **82.41** |

### Tiny ImageNet (LDA helps ✅)

| Method | R18 (512D) | R50 (2048D) | MV3 (576D) | EB0 (1280D) | ViT (768D) | DiNO (384D) |
|--------|-----------|------------|-----------|------------|-----------|------------|
| Full | 59.82 | 74.13 | 59.46 | 71.67 | 81.58 | 78.25 |
| PCA | 64.32 | 73.32 | 61.94 | 71.97 | 81.44 | **79.83** |
| **LDA** | 64.28 | 74.98 | 63.34 | 72.32 | 81.66 | 79.67 |
| DSB | **64.78** | 75.18 | **63.84** | 72.12 | **81.67** | 79.54 |

### CUB-200 (LDA hurts ❌ — boundary condition)

| Method | R18 (512D) | R50 (2048D) | MV3 (576D) | EB0 (1280D) | ViT (768D) | DiNO (384D) |
|--------|-----------|------------|-----------|------------|-----------|------------|
| **Full** | **62.89** | **64.46** | **63.01** | **78.05** | **75.72** | **87.73** |
| PCA | 55.16 | 56.93 | 56.27 | 74.06 | 73.47 | 85.80 |
| LDA | 57.73 | 57.32 | 59.30 | 75.32 | 73.61 | 85.61 |
| Best reduced | 59.39 | 61.06 | 60.75 | 76.82 | 74.51 | 86.30 |

Full 10-method tables with timing are in the paper (Tables I–III).

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
│   └── figures/                      # 5 publication figures (PDF + PNG)
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

- **180 experiments** (6 backbones × 3 datasets × 10 methods) with 5-seed averaging
- **Paired t-tests** and **Wilcoxon signed-rank tests** for all LDA vs. method comparisons
- **Pareto analysis** for accuracy–cost tradeoff (Full features: never Pareto-optimal on coarse-grained)
- **Fine-grained boundary condition**: CUB-200 establishes clear failure mode for LDA
- **Data efficiency study**: LDA crossover at ~25–50% training data
- **Component sweep**: d = C−1 confirmed optimal; monotonic improvement, no overfitting

---

## Citation

```bibtex
@article{kumar2026lda,
  title={Supervised Dimensionality Reduction Revisited: Why {LDA} on Frozen {CNN}
         Features Deserves a Second Look},
  author={Kumar, Indar and Karhana, Girish and Jasti, Sai Krishna and Lade, Ankit Hemant},
  journal={arXiv preprint arXiv:2604.03928},
  year={2026},
  url={https://arxiv.org/abs/2604.03928}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
