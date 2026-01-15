# LDA–CIFAR100: Discriminative Dimensionality Reduction Study

This project studies how classification accuracy on CIFAR-100 varies with the number of LDA components, compared against PCA and Random Projection baselines.

## Project Structure

```
lda-cifar100/
│
├── data/
│   └── load_cifar100.py          # CIFAR-100 data loading
│
├── features/
│   ├── extract_features.py       # ResNet-18 feature extraction
│   └── saved/                    # Saved features (generated)
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       └── y_test.npy
│
├── reduction/
│   ├── lda.py                    # Linear Discriminant Analysis
│   ├── pca.py                    # Principal Component Analysis
│   └── random_projection.py      # Gaussian Random Projection
│
├── models/
│   └── linear_classifier.py      # Logistic Regression classifier
│
├── experiments/
│   └── run_experiment.py         # Main experiment runner
│
├── results/
│   └── results.csv               # Experiment results (generated)
│
├── notebooks/
│   └── analysis.ipynb            # Results analysis (plots only)
│
├── figures/                      # Generated figures
│
└── README.md
```

## Requirements

```
torch
torchvision
numpy
pandas
scikit-learn
matplotlib
seaborn
tqdm
```

## Environment Setup

### Create Virtual Environment

```bash
cd lda-cifar100

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn tqdm
```

### Deactivate Environment (when done)

```bash
deactivate
```

## Usage

**Note:** Always activate the virtual environment before running commands.

```bash
source venv/bin/activate  # Activate first!
```

### Step 1: Extract Features

Extract 512-dimensional features from CIFAR-100 using frozen ResNet-18:

```bash
cd lda-cifar100
python features/extract_features.py
```

This will:
- Download CIFAR-100 (if needed)
- Extract features using ImageNet-pretrained ResNet-18
- Save features to `features/saved/`

**Note:** Features are extracted ONCE and reused for all experiments.

### Step 2: Run Experiments

Run all experiments with LDA, PCA, and Random Projection:

```bash
python experiments/run_experiment.py
```

This will:
- Load extracted features
- Test each method with components: [2, 5, 10, 20, 40, 80, 99]
- Run 5 seeds per configuration
- Save results to `results/results.csv`

### Step 3: Analyze Results

Open the analysis notebook:

```bash
jupyter notebook notebooks/analysis.ipynb
```

The notebook:
- Loads saved results
- Computes summary statistics
- Generates publication-ready plots
- Documents observed trends

## Experiment Details

### Methods
- **LDA**: Linear Discriminant Analysis (supervised, max 99 components for 100 classes)
- **PCA**: Principal Component Analysis (unsupervised)
- **RP**: Gaussian Random Projection (baseline)

### Configuration
- Feature extractor: Frozen ResNet-18 (ImageNet pretrained)
- Feature dimension: 512
- Classifier: Logistic Regression (multinomial, fixed hyperparameters)
- Component values: [2, 5, 10, 20, 40, 80, 99]
- Random seeds: [0, 1, 2, 3, 4]
- Total experiments: 3 methods × 7 component values × 5 seeds = 105

### Data
- CIFAR-100: 50,000 train / 10,000 test images
- 100 classes, 32×32 RGB images
- Standard torchvision split (no modifications)

## Results Format

Results are saved to `results/results.csv`:

| method | components | seed | accuracy |
|--------|------------|------|----------|
| lda    | 2          | 0    | 0.xxxx   |
| lda    | 2          | 1    | 0.xxxx   |
| ...    | ...        | ...  | ...      |

## Reproducibility

All experiments use fixed random seeds for reproducibility:
- NumPy random seed set before each experiment
- Classifier random_state matches experiment seed
- DataLoader shuffle=False during feature extraction

## Design Principles

1. **Scripts produce results. Notebooks only analyze.**
2. **Same classifier across all methods** (Logistic Regression)
3. **No hyperparameter tuning on test data**
4. **All results saved**, including failed runs
5. **Features extracted once**, reused for all experiments

## License

Research project - contact author for usage.
