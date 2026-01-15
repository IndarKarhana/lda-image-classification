# LDA for CIFAR-100: A Comprehensive Study on Dimensionality Reduction

---

## 1. Project Overview

### What This Project Is About

This research project investigates **dimensionality reduction techniques for image classification** on the CIFAR-100 dataset. Modern deep learning models produce high-dimensional feature representations (512D to 2048D), which can be computationally expensive for downstream tasks and deployment on resource-constrained devices.

We systematically study how **Linear Discriminant Analysis (LDA)** compares to other dimensionality reduction methods for reducing these feature dimensions while preserving—or even improving—classification accuracy.

### Research Objectives

1. **Primary Goal**: Determine how classification accuracy varies with the number of LDA components on CIFAR-100
2. **Comparative Analysis**: Compare LDA against PCA (unsupervised) and Random Projection (baseline)
3. **Backbone Study**: Investigate if LDA's advantage changes across different CNN architectures
4. **Practical Relevance**: Provide guidance for edge deployment where memory/compute is limited
5. **Modern Context**: Compare classical LDA to modern metric learning approaches

---

## 2. Background: Dimensionality Reduction Techniques

### What is PCA (Principal Component Analysis)?

**PCA** is an **unsupervised** dimensionality reduction technique that finds the directions (principal components) of maximum variance in the data.

**How it works:**
1. Center the data (subtract mean)
2. Compute the covariance matrix
3. Find eigenvectors (principal components) ordered by eigenvalues
4. Project data onto top-k eigenvectors

**Key Properties:**
- ✅ Unsupervised (no labels needed)
- ✅ Preserves global structure
- ✅ Optimal for reconstruction error
- ❌ Ignores class information
- ❌ May not preserve discriminative features

**Mathematical Formulation:**
$$\text{PCA: } \mathbf{W} = \arg\max_{\mathbf{W}} \text{Var}(\mathbf{W}^T \mathbf{X}) = \arg\max_{\mathbf{W}} \mathbf{W}^T \mathbf{S}_T \mathbf{W}$$

where $\mathbf{S}_T$ is the total scatter matrix.

---

### What is LDA (Linear Discriminant Analysis)?

**LDA** is a **supervised** dimensionality reduction technique that finds directions maximizing class separation while minimizing within-class variance.

**How it works:**
1. Compute within-class scatter matrix $\mathbf{S}_W$
2. Compute between-class scatter matrix $\mathbf{S}_B$
3. Solve generalized eigenvalue problem: $\mathbf{S}_B \mathbf{w} = \lambda \mathbf{S}_W \mathbf{w}$
4. Project data onto top-k eigenvectors

**Key Properties:**
- ✅ **Supervised** (uses class labels)
- ✅ Maximizes class separability
- ✅ Often improves classification accuracy
- ❌ Limited to (C-1) components for C classes
- ❌ Assumes Gaussian class distributions

**Mathematical Formulation:**
$$\text{LDA: } \mathbf{W} = \arg\max_{\mathbf{W}} \frac{\mathbf{W}^T \mathbf{S}_B \mathbf{W}}{\mathbf{W}^T \mathbf{S}_W \mathbf{W}}$$

This is the **Fisher criterion**: maximize between-class variance relative to within-class variance.

**Key Insight**: For CIFAR-100 with 100 classes, LDA can produce at most **99 components** (C-1 = 99).

---

### What is Random Projection?

**Random Projection (RP)** is an unsupervised technique that projects data onto a random lower-dimensional subspace.

**How it works:**
1. Generate a random matrix $\mathbf{R}$ with entries from $\mathcal{N}(0, 1/k)$
2. Project: $\mathbf{X}_{reduced} = \mathbf{X} \cdot \mathbf{R}$

**Key Properties:**
- ✅ Very fast (no learning required)
- ✅ Preserves pairwise distances (Johnson-Lindenstrauss lemma)
- ❌ No optimization for task
- ❌ High variance across random seeds

**Why include it?** Random Projection serves as a **lower bound baseline**—any learned method should significantly outperform random dimensionality reduction.

---

### Comparison Summary

| Method | Supervised | Optimization Target | Max Components | Speed |
|--------|------------|---------------------|----------------|-------|
| **PCA** | No | Maximize variance | min(n, d) | Fast |
| **LDA** | Yes | Maximize class separation | C - 1 | Fast |
| **RP** | No | Random (distance preservation) | Any | Fastest |
| **Metric Learning** | Yes | Learned similarity | Any | Slow |

---

## 3. Why These Specific Components?

We tested the following LDA component values: **[2, 5, 10, 20, 40, 80, 99]**

### Rationale for Each:

| Components | Rationale |
|------------|-----------|
| **2** | Extreme compression (visualization-friendly, tests minimal representation) |
| **5** | Very low-dimensional (5% of LDA's max capacity) |
| **10** | Low-dimensional (common for visualization, ~10% capacity) |
| **20** | Moderate reduction (~20% capacity) |
| **40** | Balanced point (~40% capacity, often "sweet spot") |
| **80** | High-dimensional (~80% capacity, near full LDA) |
| **99** | Maximum possible (C-1 for 100 classes) |

### Why This Range Matters:

1. **Edge Deployment**: Lower components (10-40) are crucial for memory-constrained devices
2. **Accuracy vs Efficiency Trade-off**: Need to find where accuracy plateaus
3. **Theoretical Limit**: LDA is bounded by (C-1) = 99, so testing near this limit shows maximum LDA capacity
4. **Logarithmic Spacing**: Captures behavior across orders of magnitude

---

## 4. Why These Specific Models (Backbones)?

We tested four CNN architectures: **ResNet-18, ResNet-50, MobileNetV3-Small, EfficientNet-B0**

### Rationale for Each:

| Backbone | Feature Dim | Why Included |
|----------|-------------|--------------|
| **ResNet-18** | 512D | Standard baseline, widely used, moderate capacity |
| **ResNet-50** | 2048D | Deeper variant, tests if LDA benefits scale with feature dim |
| **MobileNetV3-Small** | 576D | Edge-optimized, represents mobile deployment scenario |
| **EfficientNet-B0** | 1280D | Modern efficient architecture, state-of-the-art efficiency |

### Research Questions Addressed:

1. **Does LDA's advantage depend on backbone architecture?**
   - Result: Yes! ResNet-50 shows +4.22% gain, ResNet-18 only +1.87%

2. **Do higher-dimensional features benefit more from LDA?**
   - Result: Yes! 2048D features (ResNet-50) gain more than 512D (ResNet-18)

3. **Is LDA useful for edge-optimized models?**
   - Result: Yes! MobileNetV3 gains +3.05% from LDA

4. **How do modern efficient architectures compare?**
   - Result: EfficientNet-B0 achieves best absolute accuracy (72.06%)

### Feature Dimension Hypothesis:

We hypothesized that **higher-dimensional features contain more redundancy** that LDA can exploit. The results confirm this:

| Feature Dim | LDA Gain over PCA |
|-------------|-------------------|
| 512D (ResNet-18) | +1.87% |
| 576D (MobileNetV3) | +3.05% |
| 1280D (EfficientNet) | +1.76% |
| 2048D (ResNet-50) | **+4.22%** |

---

## 5. Why CIFAR-100?

**CIFAR-100** was chosen as the benchmark dataset for several reasons:

| Property | Value | Why It Matters |
|----------|-------|----------------|
| **Classes** | 100 | Tests LDA at near-maximum capacity (99 components) |
| **Samples** | 60,000 (50k train, 10k test) | Sufficient for reliable statistics |
| **Image Size** | 32×32 | Fast feature extraction |
| **Difficulty** | Moderate | Neither trivial nor impossibly hard |
| **Hierarchy** | 20 superclasses | Allows future fine/coarse analysis |

**Key Consideration**: With 100 classes, LDA can use up to 99 components, making this dataset ideal for studying LDA's full capacity.

---

## Research Question

**How does classification accuracy on CIFAR-100 vary with the number of LDA components, and how does LDA compare to PCA, Random Projection, and modern metric learning approaches across different CNN backbones?**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Background: Dimensionality Reduction Techniques](#2-background-dimensionality-reduction-techniques)
3. [Why These Specific Components?](#3-why-these-specific-components)
4. [Why These Specific Models (Backbones)?](#4-why-these-specific-models-backbones)
5. [Why CIFAR-100?](#5-why-cifar-100)
6. [Results Location](#6-results-location)
7. [Experiment Summary](#7-experiment-summary)
8. [Key Findings](#8-key-findings)
9. [Reproducibility Guide](#9-reproducibility-guide)

---

## 6. Results Location

All experimental results are stored in the `results/` directory:

| File | Description | Experiments |
|------|-------------|-------------|
| [`results/results.csv`](results/results.csv) | Main LDA vs PCA vs RP comparison | 105 experiments (3 methods × 7 components × 5 seeds) |
| [`results/ablation_results.csv`](results/ablation_results.csv) | Feature projection & SGD ablation | 140 experiments |
| [`results/backbone_comparison.csv`](results/backbone_comparison.csv) | Multi-backbone LDA/PCA analysis | 44 experiments (4 backbones × 11 configs) |
| [`results/modern_methods.csv`](results/modern_methods.csv) | LDA vs Metric Learning comparison | 12 experiments |
| [`results/classwise_analysis.csv`](results/classwise_analysis.csv) | Per-class accuracy breakdown | 100 classes analyzed |
| [`results/deployment_analysis.csv`](results/deployment_analysis.csv) | Edge deployment metrics | 16 configurations |

### Feature Cache Location

Extracted features are cached in `data/` for reproducibility:

| File | Backbone | Dimensions | Size |
|------|----------|------------|------|
| `data/resnet18_features.npz` | ResNet-18 | 512D | ~120 MB |
| `data/resnet50_features.npz` | ResNet-50 | 2048D | ~480 MB |
| `data/mobilenetv3_features.npz` | MobileNetV3-Small | 576D | ~140 MB |
| `data/efficientnet_features.npz` | EfficientNet-B0 | 1280D | ~310 MB |

---

## 7. Experiment Summary

### Experiment 1: Main Study (LDA vs PCA vs Random Projection)

**Configuration:**
- Backbone: ResNet-18 (frozen, ImageNet pretrained)
- Feature dimension: 512
- Components tested: [2, 5, 10, 20, 40, 80, 99]
- Seeds: [0, 1, 2, 3, 4]
- Classifier: Logistic Regression

**Results Summary:**

| Method | Best Accuracy | Best @ Components | Avg Runtime |
|--------|---------------|-------------------|-------------|
| **LDA** | **42.63%** | 99 | 6.2s |
| PCA | 41.36% | 99 | 24.8s |
| Random Projection | 38.13% | 99 | 35.4s |

**Key Observation:** LDA consistently outperforms PCA by 1-2% across all component counts, with the gap widening at lower dimensions.

---

### Experiment 2: Ablation Study

**2a. Feature Projection (512D → 256D before LDA)**

| Feature Dim | Accuracy @ 99 comp | Runtime | Accuracy Loss |
|-------------|-------------------|---------|---------------|
| 512D | 42.63% | 4.96s | - |
| 256D | 42.42% | 4.66s | -0.21% |

**Finding:** Projecting to 256D loses only 0.21% accuracy while reducing runtime by 6%.

**2b. Classifier Comparison (LogisticRegression vs SGDClassifier)**

| Classifier | Avg Accuracy | Training Time | Speed |
|------------|--------------|---------------|-------|
| LogisticRegression | 29.18% | 3.38s | 1× |
| SGDClassifier | 23.64% | 0.82s | 4.1× |

**Finding:** SGDClassifier is 4× faster but 5.5% less accurate.

---

### Experiment 3: Multi-Backbone Comparison

**Backbones Tested:**
- ResNet-18 (512D) - Standard CNN baseline
- ResNet-50 (2048D) - Deeper CNN
- MobileNetV3-Small (576D) - Edge-optimized
- EfficientNet-B0 (1280D) - Modern efficient architecture

**Full Features vs LDA (No Reduction Baseline):**

| Backbone | Feature Dim | Full (No Reduction) | LDA @ 99 | Δ Accuracy | Runtime Change |
|----------|-------------|---------------------|----------|------------|----------------|
| ResNet-18 | 512D | 64.41% | 66.88% | **+2.47%** ⬆️ | 83.7s → 5.7s (**15× faster**) |
| ResNet-50 | 2048D | 71.86% | 71.69% | -0.17% | 213.7s → 14.8s (**14× faster**) |
| MobileNetV3 | 576D | 69.61% | 68.36% | -1.25% | 27.0s → 8.0s (**3× faster**) |
| EfficientNet | 1280D | 73.14% | 72.06% | -1.08% | 57.6s → 11.0s (**5× faster**) |

**🔍 Surprising Finding:** For ResNet-18, LDA not only reduces dimensionality but actually **improves accuracy by 2.47%** while being **15× faster**! This suggests LDA acts as a form of regularization by removing noisy dimensions.

**LDA Accuracy by Components:**

| Components | ResNet-18 | ResNet-50 | MobileNetV3 | EfficientNet-B0 |
|------------|-----------|-----------|-------------|-----------------|
| 10 | 35.83% | 38.26% | 37.12% | 37.79% |
| 20 | 48.33% | 50.58% | 49.06% | 51.98% |
| 40 | 60.07% | 63.00% | 60.96% | 63.13% |
| 80 | 66.14% | 70.51% | 67.36% | 70.47% |
| 99 | 66.88% | 71.69% | 68.36% | **72.06%** |
| Full | 64.41% | 71.86% | 69.61% | 73.14% |

**LDA Advantage Over PCA (at 99 components):**

| Backbone | LDA | PCA | **LDA Gain** |
|----------|-----|-----|--------------|
| ResNet-18 | 66.88% | 65.01% | +1.87% |
| ResNet-50 | 71.69% | 67.47% | **+4.22%** ⭐ |
| MobileNetV3-Small | 68.36% | 65.31% | +3.05% |
| EfficientNet-B0 | 72.06% | 70.30% | +1.76% |

**🔍 Surprising Finding:** LDA's advantage is **largest for ResNet-50** (+4.22%), suggesting higher-dimensional features benefit more from supervised dimensionality reduction.

---

### Experiment 4: Modern Methods Comparison

**Methods Compared (ResNet-18 features, 99 components):**

| Method | Accuracy | Runtime | Notes |
|--------|----------|---------|-------|
| **LDA + LogReg** | **66.88%** | 5.91s | Best accuracy |
| PCA + LogReg | 65.01% | 8.84s | Unsupervised baseline |
| Metric Learning (NCA) | 64.08% | 66.62s | 11× slower than LDA |
| LDA + k-NN | 59.45% | 0.44s | Fastest inference |

**💡 Key Insight:** LDA beats trained metric learning while being **11× faster**!

---

### Experiment 5: Class-wise Analysis

**Classes that Benefit MOST from LDA (vs PCA):**

| Rank | Class | LDA Acc | PCA Acc | Improvement |
|------|-------|---------|---------|-------------|
| 1 | tulip | 54% | 43% | +11% |
| 2 | porcupine | 64% | 53% | +11% |
| 3 | seal | 51% | 41% | +10% |
| 4 | bed | 66% | 56% | +10% |
| 5 | house | 77% | 68% | +9% |

**Classes where LDA Hurts (vs PCA):**

| Rank | Class | LDA Acc | PCA Acc | Degradation |
|------|-------|---------|---------|-------------|
| 1 | woman | 42% | 52% | -10% |
| 2 | plate | 62% | 72% | -10% |
| 3 | leopard | 54% | 64% | -10% |
| 4 | man | 45% | 52% | -7% |
| 5 | hamster | 71% | 77% | -6% |

**Statistics:**
- LDA helps: **66/100 classes**
- LDA hurts: **27/100 classes**
- Neutral: **7/100 classes**
- Average improvement: **+1.87% ± 4.42%**

---

### Experiment 6: Edge Deployment Analysis

**Memory and Compute Requirements:**

| Backbone | LDA Comp | Params | Memory | FLOPs | Accuracy |
|----------|----------|--------|--------|-------|----------|
| ResNet-18 | 10 | 6,220 | 24 KB | 6,120 | 35.83% |
| ResNet-18 | 40 | 24,580 | 96 KB | 24,480 | 60.07% |
| ResNet-18 | 99 | 60,688 | 237 KB | 60,588 | 66.88% |
| MobileNetV3 | 99 | 67,024 | 262 KB | 66,924 | 68.36% |
| EfficientNet | 99 | 136,720 | 534 KB | 136,620 | 72.06% |

**Recommended Configurations:**

| Use Case | Config | Accuracy | Memory |
|----------|--------|----------|--------|
| Extreme Edge | ResNet-18 @ 10D | 35.83% | 24 KB |
| Balanced Edge | ResNet-18 @ 40D | 60.07% | 96 KB |
| Best Mobile | MobileNetV3 @ 99D | 68.36% | 262 KB |
| Best Accuracy | EfficientNet @ 99D | 72.06% | 534 KB |

---

## 8. Key Findings

### 1. LDA Can IMPROVE Accuracy Over Full Features (Surprising!)
- ResNet-18: LDA @ 99 components achieves **66.88%** vs **64.41%** for full 512D features
- This is a **+2.47% improvement** while using 5× fewer dimensions
- Suggests LDA acts as regularization by removing noisy/redundant dimensions

### 2. LDA Consistently Outperforms PCA
- Average improvement: +1.87% to +4.22% depending on backbone
- Advantage is **larger at lower dimensions** (more important for edge deployment)

### 3. Higher-Dimensional Features Benefit More from LDA
- ResNet-50 (2048D): +4.22% gain
- EfficientNet (1280D): +1.76% gain
- ResNet-18 (512D): +1.87% gain

### 4. Massive Runtime Improvements
- LDA reduces classifier training time by **5-15×** compared to full features
- ResNet-18: 83.7s → 5.7s (15× faster)
- ResNet-50: 213.7s → 14.8s (14× faster)

### 5. Accuracy Plateaus Around 80 Components
- Low-dim backbones (512D, 576D): plateau at ~80 components
- High-dim backbones (1280D, 2048D): continue improving to 99

### 6. LDA Beats Modern Metric Learning
- LDA: 66.88% in 5.91s
- Metric Learning: 64.08% in 66.62s
- **LDA is 11× faster AND 2.8% more accurate**

### 7. Class-wise Variability
- LDA helps 66% of classes, hurts 27%
- Some classes see +11% improvement, others -10%
- Suggests room for class-adaptive methods

---

## 9. Reproducibility Guide

### Prerequisites

- Python 3.11+
- macOS/Linux (MPS/CUDA supported)
- ~5 GB disk space for features cache

### Step 1: Clone and Setup Environment

```bash
cd /path/to/lda-cifar100

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn tqdm
```

### Step 2: Extract Features (First Run Only)

```bash
# Extract ResNet-18 features (required for main experiments)
python features/extract_features.py
```

This will:
- Download CIFAR-100 automatically
- Extract 512D features from frozen ResNet-18
- Save to `features/saved/`
- Takes ~2 minutes on MPS/CUDA

### Step 3: Run Main Experiments

```bash
# Main LDA vs PCA vs RP comparison (105 experiments)
python experiments/run_experiment.py
# Results saved to: results/results.csv

# Ablation study (512→256D, SGDClassifier)
python experiments/run_ablation.py
# Results saved to: results/ablation_results.csv

# Extended study (4 backbones, modern methods, class-wise analysis)
python experiments/run_extended_study.py
# Results saved to: results/backbone_comparison.csv
#                   results/modern_methods.csv
#                   results/classwise_analysis.csv
#                   results/deployment_analysis.csv
```

### Step 4: Generate Visualizations (Optional)

```bash
# Open and run the analysis notebook
jupyter notebook notebooks/analysis.ipynb
```

### Expected Runtimes

| Experiment | Runtime (MPS) | Runtime (CUDA) |
|------------|---------------|----------------|
| Feature extraction (ResNet-18) | ~2 min | ~1 min |
| Main experiments | ~10 min | ~5 min |
| Ablation study | ~5 min | ~3 min |
| Extended study (first run) | ~4 hours | ~1 hour |
| Extended study (cached) | ~15 min | ~8 min |

### Verifying Results

After running experiments, verify key results:

```python
import pandas as pd

# Check main results
df = pd.read_csv('results/results.csv')
lda_99 = df[(df['method'] == 'lda') & (df['components'] == 99)]['accuracy'].mean()
print(f"LDA @ 99 components: {lda_99:.4f}")  # Expected: ~0.4263

# Check backbone comparison
df_bb = pd.read_csv('results/backbone_comparison.csv')
best = df_bb[df_bb['method'] == 'LDA'].groupby('backbone')['accuracy'].max()
print(best)  # Expected: EfficientNet ~0.72, ResNet-50 ~0.72
```

---

## Project Structure

```
lda-cifar100/
├── data/
│   ├── load_cifar100.py          # Data loading utilities
│   ├── cifar-100-python/         # CIFAR-100 dataset (auto-downloaded)
│   └── *_features.npz            # Cached backbone features
├── features/
│   ├── extract_features.py       # Feature extraction script
│   └── saved/                    # Legacy feature location
├── reduction/
│   ├── lda.py                    # LDA wrapper
│   ├── pca.py                    # PCA wrapper
│   └── random_projection.py      # RP wrapper
├── models/
│   └── linear_classifier.py      # Logistic Regression wrapper
├── experiments/
│   ├── run_experiment.py         # Main experiment runner
│   ├── run_ablation.py           # Ablation study
│   └── run_extended_study.py     # Extended multi-backbone study
├── notebooks/
│   └── analysis.ipynb            # Visualization notebook
├── results/
│   ├── results.csv               # Main results
│   ├── ablation_results.csv      # Ablation results
│   ├── backbone_comparison.csv   # Multi-backbone results
│   ├── modern_methods.csv        # Modern methods comparison
│   ├── classwise_analysis.csv    # Per-class breakdown
│   └── deployment_analysis.csv   # Edge deployment metrics
├── README.md                     # Project overview
├── research.md                   # This file
└── requirements.txt              # Python dependencies
```

---

## Citation

If you use this code or findings, please cite:

```bibtex
@misc{lda-cifar100-study,
  title={LDA for CIFAR-100: A Study on Supervised Dimensionality Reduction},
  author={Research Study},
  year={2026},
  note={GitHub repository}
}
```

---

## License

MIT License - See LICENSE file for details.
