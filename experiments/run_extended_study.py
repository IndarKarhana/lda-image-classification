"""
Extended Study: Multi-Backbone Comparison, Class-wise Analysis, Modern Methods
================================================================================
1. Multiple CNN backbones - ResNet-18, ResNet-50, MobileNetV3, EfficientNet-B0
2. Class-wise accuracy breakdown - Which classes benefit most from LDA?
3. Modern baselines - Supervised Contrastive, Metric Learning approaches
4. Edge deployment framing - Memory/compute analysis

Author: Research Study
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from torchvision.datasets import CIFAR100

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, normalize

import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# PART 1: Feature Extraction for Multiple Backbones
# ============================================================================

def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class FeatureExtractor:
    """Generic feature extractor for torchvision models."""
    
    def __init__(self, model_name: str, device: torch.device):
        self.model_name = model_name
        self.device = device
        self.model, self.feature_dim = self._load_model()
        self.model.eval()
        self.model.to(device)
    
    def _load_model(self):
        """Load model and return (model, feature_dim)."""
        if self.model_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            feature_dim = 512
            model.fc = nn.Identity()
        elif self.model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            feature_dim = 2048
            model.fc = nn.Identity()
        elif self.model_name == 'mobilenetv3':
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            feature_dim = 576
            model.classifier = nn.Identity()
        elif self.model_name == 'efficientnet':
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            feature_dim = 1280
            model.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        return model, feature_dim
    
    @torch.no_grad()
    def extract(self, dataloader):
        """Extract features from dataloader."""
        features = []
        labels = []
        
        for images, targets in dataloader:
            images = images.to(self.device)
            feats = self.model(images)
            features.append(feats.cpu().numpy())
            labels.append(targets.numpy())
        
        return np.vstack(features), np.concatenate(labels)


def extract_backbone_features(backbone_name: str, data_dir: Path, device: torch.device) -> dict:
    """Extract features for a specific backbone."""
    
    # Map display name to model name
    model_map = {
        'ResNet-18': 'resnet18',
        'ResNet-50': 'resnet50', 
        'MobileNetV3-Small': 'mobilenetv3',
        'EfficientNet-B0': 'efficientnet'
    }
    
    model_name = model_map[backbone_name]
    save_path = data_dir / f"{model_name}_features.npz"
    
    # Check cache
    if save_path.exists():
        print(f"  [{backbone_name}] Loading cached features...")
        data = np.load(save_path)
        X_train = data['X_train']
        # Get dim from data or infer from shape
        if 'dim' in data.files:
            dim = int(data['dim'])
        else:
            dim = X_train.shape[1]
        return {
            'X_train': X_train,
            'y_train': data['y_train'],
            'X_test': data['X_test'],
            'y_test': data['y_test'],
            'dim': dim
        }
    
    print(f"  [{backbone_name}] Extracting features...")
    start_time = time.perf_counter()
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = CIFAR100(root=str(data_dir), train=True, 
                             download=True, transform=transform)
    test_dataset = CIFAR100(root=str(data_dir), train=False,
                            download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, 
                              shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128,
                             shuffle=False, num_workers=0)
    
    extractor = FeatureExtractor(model_name, device)
    
    X_train, y_train = extractor.extract(train_loader)
    X_test, y_test = extractor.extract(test_loader)
    
    elapsed = time.perf_counter() - start_time
    print(f"    Extracted in {elapsed:.1f}s | Shape: {X_train.shape}")
    
    # Cache features
    np.savez(save_path, X_train=X_train, y_train=y_train,
             X_test=X_test, y_test=y_test, dim=extractor.feature_dim)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'dim': extractor.feature_dim
    }


# ============================================================================
# PART 2: Dimension Reduction Methods
# ============================================================================

def apply_lda(X_train, y_train, X_test, n_components):
    """Apply LDA reduction."""
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    return X_train_lda, X_test_lda

def apply_pca(X_train, X_test, n_components):
    """Apply PCA reduction."""
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca


# ============================================================================
# PART 3: Modern Baseline Methods
# ============================================================================

class NCALoss(nn.Module):
    """Neighborhood Component Analysis loss for metric learning."""
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, labels):
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature
        
        # Create mask for positive pairs
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.t()).float()
        
        # Remove diagonal
        mask = mask - torch.eye(mask.size(0), device=mask.device)
        
        # Compute NCA loss
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Mean of positive pairs
        mean_log_prob = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        loss = -mean_log_prob.mean()
        
        return loss


class MetricLearner(nn.Module):
    """Simple linear projection for metric learning."""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.projection(x)


def train_metric_learning(X_train, y_train, X_test, output_dim, 
                         epochs=50, batch_size=256, lr=0.01):
    """Train metric learning projection."""
    device = get_device()
    
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = MetricLearner(X_train.shape[1], output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = NCALoss(temperature=0.1)
    
    # Train
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            embeddings = model(batch_x)
            loss = criterion(embeddings, batch_y)
            loss.backward()
            optimizer.step()
    
    # Transform
    model.eval()
    with torch.no_grad():
        X_train_proj = model(torch.FloatTensor(X_train).to(device)).cpu().numpy()
        X_test_proj = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()
    
    return X_train_proj, X_test_proj


def train_supervised_contrastive(X_train, y_train, X_test, output_dim,
                                 epochs=50, batch_size=256, lr=0.01):
    """Supervised contrastive learning projection (SupCon simplified)."""
    return train_metric_learning(X_train, y_train, X_test, output_dim,
                                epochs, batch_size, lr)


# ============================================================================
# PART 4: Classifiers
# ============================================================================

def train_and_evaluate(X_train, y_train, X_test, y_test, method='logistic'):
    """Train classifier and return accuracy + predictions."""
    if method == 'logistic':
        clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    elif method == 'knn':
        clf = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy, y_pred


# ============================================================================
# PART 5: Class-wise Analysis
# ============================================================================

CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]


def compute_classwise_accuracy(y_true, y_pred, n_classes=100):
    """Compute per-class accuracy."""
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for true, pred in zip(y_true, y_pred):
        class_total[true] += 1
        if true == pred:
            class_correct[true] += 1
    
    accuracies = {}
    for c in range(n_classes):
        if class_total[c] > 0:
            accuracies[c] = class_correct[c] / class_total[c]
        else:
            accuracies[c] = 0.0
    
    return accuracies


def analyze_lda_benefit(baseline_classwise, lda_classwise, n_classes=100):
    """Analyze which classes benefit most from LDA."""
    improvements = {}
    for c in range(n_classes):
        improvements[c] = lda_classwise[c] - baseline_classwise[c]
    
    # Sort by improvement
    sorted_improvements = sorted(improvements.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_improvements


# ============================================================================
# PART 6: Edge Deployment Analysis
# ============================================================================

def compute_deployment_metrics(feature_dim, n_components, n_train=50000, n_classes=100):
    """Compute memory and compute requirements for edge deployment."""
    
    # LDA projection matrix: feature_dim x n_components
    lda_params = feature_dim * n_components
    
    # Logistic regression: n_components x n_classes + n_classes (bias)
    lr_params = n_components * n_classes + n_classes
    
    # Total parameters
    total_params = lda_params + lr_params
    
    # Memory (assuming float32 = 4 bytes)
    memory_mb = (total_params * 4) / (1024 * 1024)
    
    # Inference FLOPs per sample
    # LDA: feature_dim * n_components (matrix multiply)
    # LR: n_components * n_classes
    inference_flops = feature_dim * n_components + n_components * n_classes
    
    return {
        'lda_params': lda_params,
        'classifier_params': lr_params,
        'total_params': total_params,
        'memory_mb': memory_mb,
        'inference_flops': inference_flops
    }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_extended_study():
    """Run the complete extended study."""
    
    device = get_device()
    print(f"Using device: {device}")
    
    data_dir = project_root / "data"
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 70)
    print("EXTENDED STUDY: Multi-Backbone + Modern Methods + Class Analysis")
    print("=" * 70)
    
    # ========================================================================
    # SECTION 1: Extract features from all backbones
    # ========================================================================
    print("\n" + "=" * 70)
    print("SECTION 1: Feature Extraction (4 CNN Backbones)")
    print("=" * 70)
    
    backbones = ['ResNet-18', 'ResNet-50', 'MobileNetV3-Small', 'EfficientNet-B0']
    backbone_data = {}
    
    for backbone in backbones:
        backbone_data[backbone] = extract_backbone_features(backbone, data_dir, device)
    
    print("\n  Feature dimensions:")
    for name, data in backbone_data.items():
        print(f"    {name}: {data['dim']}D")
    
    # ========================================================================
    # SECTION 2: Backbone Comparison (LDA vs PCA across backbones)
    # ========================================================================
    print("\n" + "=" * 70)
    print("SECTION 2: Backbone Comparison (LDA vs PCA)")
    print("=" * 70)
    
    components_list = [10, 20, 40, 80, 99]
    backbone_results = []
    
    for backbone_name, data in backbone_data.items():
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        feature_dim = data['dim']
        
        print(f"\n  [{backbone_name}] ({feature_dim}D)")
        
        # Baseline: No reduction
        start = time.perf_counter()
        acc_baseline, _ = train_and_evaluate(X_train, y_train, X_test, y_test)
        baseline_time = time.perf_counter() - start
        
        backbone_results.append({
            'backbone': backbone_name,
            'feature_dim': feature_dim,
            'method': 'None (Full)',
            'components': feature_dim,
            'accuracy': acc_baseline,
            'runtime_sec': baseline_time
        })
        print(f"    Full ({feature_dim}D): {acc_baseline:.4f}")
        
        for n_comp in components_list:
            # LDA
            start = time.perf_counter()
            X_train_lda, X_test_lda = apply_lda(X_train, y_train, X_test, n_comp)
            acc_lda, _ = train_and_evaluate(X_train_lda, y_train, X_test_lda, y_test)
            lda_time = time.perf_counter() - start
            
            backbone_results.append({
                'backbone': backbone_name,
                'feature_dim': feature_dim,
                'method': 'LDA',
                'components': n_comp,
                'accuracy': acc_lda,
                'runtime_sec': lda_time
            })
            
            # PCA
            start = time.perf_counter()
            X_train_pca, X_test_pca = apply_pca(X_train, X_test, n_comp)
            acc_pca, _ = train_and_evaluate(X_train_pca, y_train, X_test_pca, y_test)
            pca_time = time.perf_counter() - start
            
            backbone_results.append({
                'backbone': backbone_name,
                'feature_dim': feature_dim,
                'method': 'PCA',
                'components': n_comp,
                'accuracy': acc_pca,
                'runtime_sec': pca_time
            })
            
            lda_gain = acc_lda - acc_pca
            print(f"    {n_comp:3d}D: LDA={acc_lda:.4f} | PCA={acc_pca:.4f} | Δ={lda_gain:+.4f}")
    
    # Save backbone results
    df_backbone = pd.DataFrame(backbone_results)
    df_backbone.to_csv(results_dir / "backbone_comparison.csv", index=False)
    
    # ========================================================================
    # SECTION 3: Modern Methods Comparison
    # ========================================================================
    print("\n" + "=" * 70)
    print("SECTION 3: Modern Methods Comparison")
    print("=" * 70)
    
    # Use ResNet-18 for this comparison
    data = backbone_data['ResNet-18']
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    modern_results = []
    components_test = [20, 40, 99]
    
    for n_comp in components_test:
        print(f"\n  Components: {n_comp}")
        
        # LDA
        start = time.perf_counter()
        X_train_lda, X_test_lda = apply_lda(X_train, y_train, X_test, n_comp)
        acc_lda, _ = train_and_evaluate(X_train_lda, y_train, X_test_lda, y_test)
        lda_time = time.perf_counter() - start
        modern_results.append({
            'method': 'LDA',
            'components': n_comp,
            'accuracy': acc_lda,
            'runtime_sec': lda_time
        })
        print(f"    LDA: {acc_lda:.4f} ({lda_time:.2f}s)")
        
        # PCA
        start = time.perf_counter()
        X_train_pca, X_test_pca = apply_pca(X_train, X_test, n_comp)
        acc_pca, _ = train_and_evaluate(X_train_pca, y_train, X_test_pca, y_test)
        pca_time = time.perf_counter() - start
        modern_results.append({
            'method': 'PCA',
            'components': n_comp,
            'accuracy': acc_pca,
            'runtime_sec': pca_time
        })
        print(f"    PCA: {acc_pca:.4f} ({pca_time:.2f}s)")
        
        # Metric Learning (NCA-style)
        print(f"    Training Metric Learning...")
        start = time.perf_counter()
        X_train_ml, X_test_ml = train_metric_learning(
            X_train, y_train, X_test, n_comp, epochs=30
        )
        acc_ml, _ = train_and_evaluate(X_train_ml, y_train, X_test_ml, y_test)
        ml_time = time.perf_counter() - start
        modern_results.append({
            'method': 'Metric Learning (NCA)',
            'components': n_comp,
            'accuracy': acc_ml,
            'runtime_sec': ml_time
        })
        print(f"    Metric Learning: {acc_ml:.4f} ({ml_time:.2f}s)")
        
        # k-NN on LDA features (another baseline)
        start = time.perf_counter()
        acc_knn, _ = train_and_evaluate(X_train_lda, y_train, X_test_lda, y_test, method='knn')
        knn_time = time.perf_counter() - start
        modern_results.append({
            'method': 'LDA + k-NN',
            'components': n_comp,
            'accuracy': acc_knn,
            'runtime_sec': knn_time
        })
        print(f"    LDA + k-NN: {acc_knn:.4f} ({knn_time:.2f}s)")
    
    df_modern = pd.DataFrame(modern_results)
    df_modern.to_csv(results_dir / "modern_methods.csv", index=False)
    
    # ========================================================================
    # SECTION 4: Class-wise Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("SECTION 4: Class-wise Analysis (Which classes benefit from LDA?)")
    print("=" * 70)
    
    # Compare at 99 components (best LDA)
    X_train_lda, X_test_lda = apply_lda(X_train, y_train, X_test, 99)
    X_train_pca, X_test_pca = apply_pca(X_train, X_test, 99)
    
    _, y_pred_lda = train_and_evaluate(X_train_lda, y_train, X_test_lda, y_test)
    _, y_pred_pca = train_and_evaluate(X_train_pca, y_train, X_test_pca, y_test)
    
    classwise_lda = compute_classwise_accuracy(y_test, y_pred_lda)
    classwise_pca = compute_classwise_accuracy(y_test, y_pred_pca)
    
    improvements = analyze_lda_benefit(classwise_pca, classwise_lda)
    
    print("\n  Top 10 classes that BENEFIT from LDA (vs PCA):")
    for i, (class_id, improvement) in enumerate(improvements[:10]):
        class_name = CIFAR100_CLASSES[class_id]
        lda_acc = classwise_lda[class_id]
        pca_acc = classwise_pca[class_id]
        print(f"    {i+1:2d}. {class_name:15s}: LDA={lda_acc:.2f} PCA={pca_acc:.2f} Δ={improvement:+.2f}")
    
    print("\n  Top 10 classes where LDA HURTS (vs PCA):")
    for i, (class_id, improvement) in enumerate(improvements[-10:][::-1]):
        class_name = CIFAR100_CLASSES[class_id]
        lda_acc = classwise_lda[class_id]
        pca_acc = classwise_pca[class_id]
        print(f"    {i+1:2d}. {class_name:15s}: LDA={lda_acc:.2f} PCA={pca_acc:.2f} Δ={improvement:+.2f}")
    
    # Save class-wise results
    classwise_results = []
    for class_id in range(100):
        classwise_results.append({
            'class_id': class_id,
            'class_name': CIFAR100_CLASSES[class_id],
            'lda_accuracy': classwise_lda[class_id],
            'pca_accuracy': classwise_pca[class_id],
            'lda_improvement': classwise_lda[class_id] - classwise_pca[class_id]
        })
    
    df_classwise = pd.DataFrame(classwise_results)
    df_classwise.to_csv(results_dir / "classwise_analysis.csv", index=False)
    
    # ========================================================================
    # SECTION 5: Edge Deployment Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("SECTION 5: Edge Deployment Analysis")
    print("=" * 70)
    
    deployment_results = []
    
    for backbone_name in backbones:
        feature_dim = backbone_data[backbone_name]['dim']
        
        print(f"\n  [{backbone_name}] (Feature dim: {feature_dim})")
        
        for n_comp in [10, 20, 40, 99]:
            metrics = compute_deployment_metrics(feature_dim, n_comp)
            
            # Get accuracy from backbone results
            acc_row = df_backbone[
                (df_backbone['backbone'] == backbone_name) & 
                (df_backbone['method'] == 'LDA') & 
                (df_backbone['components'] == n_comp)
            ]
            accuracy = acc_row['accuracy'].values[0] if len(acc_row) > 0 else 0
            
            deployment_results.append({
                'backbone': backbone_name,
                'feature_dim': feature_dim,
                'lda_components': n_comp,
                'total_params': metrics['total_params'],
                'memory_kb': metrics['memory_mb'] * 1024,
                'inference_flops': metrics['inference_flops'],
                'accuracy': accuracy
            })
            
            print(f"    {n_comp:3d}D: params={metrics['total_params']:,} | "
                  f"mem={metrics['memory_mb']*1024:.1f}KB | "
                  f"FLOPs={metrics['inference_flops']:,} | "
                  f"acc={accuracy:.4f}")
    
    df_deployment = pd.DataFrame(deployment_results)
    df_deployment.to_csv(results_dir / "deployment_analysis.csv", index=False)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXTENDED STUDY SUMMARY")
    print("=" * 70)
    
    print("\n1. BACKBONE COMPARISON (at 99 LDA components):")
    print("-" * 50)
    for backbone in backbones:
        row = df_backbone[
            (df_backbone['backbone'] == backbone) & 
            (df_backbone['method'] == 'LDA') & 
            (df_backbone['components'] == 99)
        ]
        pca_row = df_backbone[
            (df_backbone['backbone'] == backbone) & 
            (df_backbone['method'] == 'PCA') & 
            (df_backbone['components'] == 99)
        ]
        if len(row) > 0:
            lda_acc = row['accuracy'].values[0]
            pca_acc = pca_row['accuracy'].values[0] if len(pca_row) > 0 else 0
            gain = lda_acc - pca_acc
            print(f"  {backbone:20s}: LDA={lda_acc:.4f} PCA={pca_acc:.4f} Δ={gain:+.4f}")
    
    print("\n2. MODERN METHODS (at 99 components, ResNet-18):")
    print("-" * 50)
    for method in df_modern['method'].unique():
        row = df_modern[(df_modern['method'] == method) & (df_modern['components'] == 99)]
        if len(row) > 0:
            print(f"  {method:25s}: {row['accuracy'].values[0]:.4f}")
    
    print("\n3. CLASS-WISE INSIGHTS:")
    print("-" * 50)
    avg_improvement = df_classwise['lda_improvement'].mean()
    std_improvement = df_classwise['lda_improvement'].std()
    classes_benefit = (df_classwise['lda_improvement'] > 0).sum()
    classes_hurt = (df_classwise['lda_improvement'] < 0).sum()
    print(f"  Average LDA improvement: {avg_improvement:+.4f} ± {std_improvement:.4f}")
    print(f"  Classes where LDA helps: {classes_benefit}/100")
    print(f"  Classes where LDA hurts: {classes_hurt}/100")
    
    print("\n4. EDGE DEPLOYMENT (Best accuracy/memory tradeoff):")
    print("-" * 50)
    # Find pareto optimal
    best_ratio = 0
    best_config = None
    for _, row in df_deployment.iterrows():
        ratio = row['accuracy'] / (row['memory_kb'] / 1024)  # acc per MB
        if ratio > best_ratio:
            best_ratio = ratio
            best_config = row
    
    if best_config is not None:
        print(f"  Best efficiency: {best_config['backbone']} @ {int(best_config['lda_components'])} components")
        print(f"    Accuracy: {best_config['accuracy']:.4f}")
        print(f"    Memory: {best_config['memory_kb']:.1f} KB")
        print(f"    Params: {int(best_config['total_params']):,}")
    
    print("\n" + "=" * 70)
    print("Results saved to:")
    print(f"  - {results_dir / 'backbone_comparison.csv'}")
    print(f"  - {results_dir / 'modern_methods.csv'}")
    print(f"  - {results_dir / 'classwise_analysis.csv'}")
    print(f"  - {results_dir / 'deployment_analysis.csv'}")
    print("=" * 70)
    
    return {
        'backbone': df_backbone,
        'modern': df_modern,
        'classwise': df_classwise,
        'deployment': df_deployment
    }


if __name__ == "__main__":
    results = run_extended_study()
