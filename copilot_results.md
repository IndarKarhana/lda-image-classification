# ✅ Project Created: lda-cifar100

## Project Summary

| File | Purpose |
|------|---------|
| `data/load_cifar100.py` | Loads CIFAR-100 with ImageNet normalization |
| `features/extract_features.py` | Frozen ResNet-18 → 512-dim features |
| `reduction/lda.py` | LDA (supervised, max 99 components) |
| `reduction/pca.py` | PCA (unsupervised baseline) |
| `reduction/random_projection.py` | Gaussian RP (baseline) |
| `models/linear_classifier.py` | Logistic Regression (fixed config) |
| `experiments/run_experiment.py` | Runs 105 experiments, saves CSV |
| `notebooks/analysis.ipynb` | Plots & statistics only |
| `README.md` | Documentation |

---

## How to Run

```bash
cd /Users/indarkumar/Documents/Research_01/lda-cifar100

# Step 1: Extract features (run once, ~5-10 min on Mac)
python features/extract_features.py

# Step 2: Run experiments (~15-30 min)
python experiments/run_experiment.py

# Step 3: Analyze in notebook
jupyter notebook notebooks/analysis.ipynb
```

---

## Guardrails Enforced

All guardrails from your guidelines are enforced:

- ✅ **Scripts produce results, notebooks only analyze**
- ✅ **Same classifier across all methods**
- ✅ **Features extracted once, reused**
- ✅ **5 seeds, all results logged**
- ✅ **No train/test leakage**
