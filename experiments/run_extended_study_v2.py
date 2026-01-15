"""
Extended Study V2: Multi-Backbone CNN Comparison + Class Analysis + Modern Methods
===================================================================================
1. Multiple CNN backbones: ResNet-18, ResNet-50, MobileNetV3, EfficientNet-B0
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


# CIFAR-100 class names
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

# Superclass groupings for CIFAR-100
SUPERCLASSES = {
    'aquatic_mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food_containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit_vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household_electrical': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    'household_furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large_carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large_omnivores_herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium_mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non_insect_invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small_mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    'vehicles_1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles_2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
    'large_natural_scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large_man_made': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
}


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ============================================================================
# BACKBONE DEFINITIONS
# ============================================================================

BACKBONES = {
    'resnet18': {
        'model_fn': lambda: models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1),
        'feature_dim': 512,
        'layer': 'avgpool',
        'description': 'Standard CNN (18 layers)'
    },
    'resnet50': {
        'model_fn': lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2),
        'feature_dim': 2048,
        'layer': 'avgpool',
        'description': 'Deep CNN (50 layers)'
    },
    'mobilenetv3_small': {
        'model_fn': lambda: models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
        'feature_dim': 576,
        'layer': 'avgpool',
        'description': 'Efficient/Edge-focused'
    },
    'efficientnet_b0': {
        'model_fn': lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1),
        'feature_dim': 1280,
        'layer': 'avgpool',
        'description': 'Modern efficient architecture'
    }
}


def extract_features_generic(backbone_name: str, data_dir: Path, device: torch.device) -> dict:
    """Extract features using any backbone."""
    config = BACKBONES[backbone_name]
    print(f"\n[{backbone_name}] Extracting features ({config['description']})...")
    
    # Check cache
    save_path = data_dir / f"{backbone_name}_features.npz"
    if save_path.exists():
        print("  Loading cached features...")
        data = np.load(save_path)
        return {
            'X_train': data['X_train'],
            'y_train': data['y_train'],
            'X_test': data['X_test'],
            'y_test': data['y_test'],
            'dim': config['feature_dim'],
            'backbone': backbone_name
        }
    
    # Setup
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
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    
    # Load model
    model = config['model_fn']()
    
    # Create feature extractor based on architecture
    if 'resnet' in backbone_name:
        model.fc = nn.Identity()
    elif 'mobilenet' in backbone_name:
        model.classifier = nn.Identity()
    elif 'efficientnet' in backbone_name:
        model.classifier = nn.Identity()
    
    model = model.to(device)
    model.eval()
    
    def extract(loader):
        features, labels = [], []
        with torch.no_grad():
            for i, (imgs, lbls) in enumerate(loader):
                imgs = imgs.to(device)
                feats = model(imgs)
                feats = feats.view(feats.size(0), -1)  # Flatten
                features.append(feats.cpu().numpy())
                labels.append(lbls.numpy())
                if (i + 1) % 50 == 0:
                    print(f"    Batch {i+1}/{len(loader)}")
        return np.vstack(features), np.concatenate(labels)
    
    start = time.perf_counter()
    X_train, y_train = extract(train_loader)
    X_test, y_test = extract(test_loader)
    elapsed = time.perf_counter() - start
    
    print(f"  Extracted in {elapsed:.1f}s | Shape: train={X_train.shape}, test={X_test.shape}")
    
    # Cache
    np.savez(save_path, X_train=X_train, y_train=y_train, 
             X_test=X_test, y_test=y_test)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'dim': config['feature_dim'],
        'backbone': backbone_name
    }


# ============================================================================
# DIMENSIONALITY REDUCTION METHODS
# ============================================================================

def run_reduction_experiment(X_train, y_train, X_test, y_test,
                             method: str, n_components: int) -> dict:
    """Run a single reduction + classification experiment."""
    results = {}
    
    # Reduction
    start = time.perf_counter()
    if method == 'lda':
        reducer = LinearDiscriminantAnalysis(n_components=min(n_components, 99))
        X_train_red = reducer.fit_transform(X_train, y_train)
        X_test_red = reducer.transform(X_test)
    elif method == 'pca':
        reducer = PCA(n_components=n_components)
        X_train_red = reducer.fit_transform(X_train)
        X_test_red = reducer.transform(X_test)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    results['reduction_sec'] = time.perf_counter() - start
    
    # Classification
    start = time.perf_counter()
    clf = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
    clf.fit(X_train_red, y_train)
    results['train_sec'] = time.perf_counter() - start
    
    # Evaluation
    y_pred = clf.predict(X_test_red)
    results['accuracy'] = accuracy_score(y_test, y_pred)
    results['predictions'] = y_pred
    
    return results


# ============================================================================
# MODERN METHODS: Supervised Contrastive & Metric Learning Proxies
# ============================================================================

class NCALoss(nn.Module):
    """Neighborhood Component Analysis loss for metric learning."""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)
        sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature
        
        # Mask for same-class pairs
        labels = labels.view(-1, 1)
        mask = (labels == labels.t()).float()
        mask.fill_diagonal_(0)
        
        # Softmax over similarities
        exp_sim = torch.exp(sim_matrix)
        exp_sim.fill_diagonal_(0)
        
        # NCA loss
        prob = (exp_sim * mask).sum(1) / exp_sim.sum(1).clamp(min=1e-8)
        loss = -torch.log(prob.clamp(min=1e-8)).mean()
        
        return loss


def train_metric_learning(X_train, y_train, output_dim: int, 
                          epochs: int = 50, batch_size: int = 256) -> nn.Module:
    """Train a simple metric learning projection."""
    device = get_device()
    input_dim = X_train.shape[1]
    
    # Simple MLP projection
    model = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, output_dim)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = NCALoss(temperature=0.1)
    
    dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            
            optimizer.zero_grad()
            embeddings = model(feats)
            loss = criterion(embeddings, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    return model


def run_metric_learning_experiment(X_train, y_train, X_test, y_test,
                                   n_components: int) -> dict:
    """Run metric learning baseline."""
    device = get_device()
    results = {}
    
    # Train projection
    start = time.perf_counter()
    model = train_metric_learning(X_train, y_train, n_components, epochs=30)
    results['train_sec'] = time.perf_counter() - start
    
    # Project features
    model.eval()
    with torch.no_grad():
        X_train_proj = model(torch.FloatTensor(X_train).to(device)).cpu().numpy()
        X_test_proj = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()
    
    # KNN classification (metric learning typically uses KNN)
    start = time.perf_counter()
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn.fit(X_train_proj, y_train)
    y_pred = knn.predict(X_test_proj)
    results['eval_sec'] = time.perf_counter() - start
    
    results['accuracy'] = accuracy_score(y_test, y_pred)
    results['predictions'] = y_pred
    
    return results


def run_supervised_contrastive_proxy(X_train, y_train, X_test, y_test,
                                     n_components: int) -> dict:
    """
    Supervised contrastive proxy: LDA + cosine similarity + KNN
    This mimics what SupCon learns: class-discriminative + normalized embeddings
    """
    results = {}
    
    # LDA reduction
    start = time.perf_counter()
    lda = LinearDiscriminantAnalysis(n_components=min(n_components, 99))
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    
    # L2 normalize (like contrastive methods)
    X_train_norm = normalize(X_train_lda)
    X_test_norm = normalize(X_test_lda)
    results['reduction_sec'] = time.perf_counter() - start
    
    # KNN with cosine similarity
    start = time.perf_counter()
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn.fit(X_train_norm, y_train)
    y_pred = knn.predict(X_test_norm)
    results['eval_sec'] = time.perf_counter() - start
    
    results['accuracy'] = accuracy_score(y_test, y_pred)
    results['predictions'] = y_pred
    
    return results


# ============================================================================
# CLASS-WISE ANALYSIS
# ============================================================================

def compute_class_accuracies(y_true, y_pred) -> dict:
    """Compute per-class accuracies."""
    class_accs = {}
    for c in range(100):
        mask = (y_true == c)
        if mask.sum() > 0:
            class_accs[c] = (y_pred[mask] == y_true[mask]).mean()
    return class_accs


def analyze_lda_class_improvement(results_lda, results_pca, y_test) -> pd.DataFrame:
    """Analyze which classes benefit most from LDA vs PCA."""
    lda_class_acc = compute_class_accuracies(y_test, results_lda['predictions'])
    pca_class_acc = compute_class_accuracies(y_test, results_pca['predictions'])
    
    data = []
    for c in range(100):
        improvement = lda_class_acc.get(c, 0) - pca_class_acc.get(c, 0)
        
        # Find superclass
        superclass = 'unknown'
        for sc, classes in SUPERCLASSES.items():
            if CIFAR100_CLASSES[c] in classes:
                superclass = sc
                break
        
        data.append({
            'class_id': c,
            'class_name': CIFAR100_CLASSES[c],
            'superclass': superclass,
            'lda_acc': lda_class_acc.get(c, 0),
            'pca_acc': pca_class_acc.get(c, 0),
            'improvement': improvement
        })
    
    return pd.DataFrame(data)


# ============================================================================
# EDGE DEPLOYMENT ANALYSIS
# ============================================================================

def compute_deployment_metrics(feature_dim: int, reduced_dim: int, 
                               n_train: int = 50000) -> dict:
    """Compute memory and compute requirements for edge deployment."""
    
    # Memory for storing reduced embeddings (float32)
    bytes_per_float = 4
    
    # Original embedding storage
    original_memory_mb = (n_train * feature_dim * bytes_per_float) / (1024 * 1024)
    
    # Reduced embedding storage
    reduced_memory_mb = (n_train * reduced_dim * bytes_per_float) / (1024 * 1024)
    
    # LDA projection matrix size
    lda_matrix_mb = (feature_dim * reduced_dim * bytes_per_float) / (1024 * 1024)
    
    # Classifier weights (100 classes)
    classifier_mb = (reduced_dim * 100 * bytes_per_float) / (1024 * 1024)
    
    # Inference FLOPs per sample
    # Projection: feature_dim * reduced_dim multiplications
    # Classification: reduced_dim * 100 multiplications
    projection_flops = feature_dim * reduced_dim
    classifier_flops = reduced_dim * 100
    total_flops = projection_flops + classifier_flops
    
    return {
        'original_memory_mb': original_memory_mb,
        'reduced_memory_mb': reduced_memory_mb,
        'memory_reduction': original_memory_mb / reduced_memory_mb,
        'lda_matrix_mb': lda_matrix_mb,
        'classifier_mb': classifier_mb,
        'total_model_mb': lda_matrix_mb + classifier_mb,
        'projection_flops': projection_flops,
        'classifier_flops': classifier_flops,
        'total_flops': total_flops,
        'flops_reduction': (feature_dim * 100) / total_flops
    }


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def run_extended_study():
    """Run the complete extended study."""
    device = get_device()
    print(f"Using device: {device}")
    
    data_dir = project_root / "data"
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("EXTENDED STUDY V2: Multi-CNN Backbone + Modern Methods + Class Analysis")
    print("=" * 70)
    
    # ========================================================================
    # SECTION 1: Extract features from all backbones
    # ========================================================================
    print("\n" + "=" * 70)
    print("SECTION 1: Multi-Backbone Feature Extraction")
    print("=" * 70)
    
    backbone_data = {}
    for backbone_name in BACKBONES.keys():
        backbone_data[backbone_name] = extract_features_generic(
            backbone_name, data_dir, device
        )
    
    # ========================================================================
    # SECTION 2: LDA vs PCA across backbones
    # ========================================================================
    print("\n" + "=" * 70)
    print("SECTION 2: LDA vs PCA Across Backbones")
    print("=" * 70)
    
    components_list = [10, 20, 40, 80, 99]
    backbone_results = []
    
    for backbone_name, data in backbone_data.items():
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        feat_dim = data['dim']
        
        print(f"\n[{backbone_name}] Feature dim: {feat_dim}")
        
        for n_comp in components_list:
            for method in ['lda', 'pca']:
                result = run_reduction_experiment(
                    X_train, y_train, X_test, y_test,
                    method=method, n_components=n_comp
                )
                
                backbone_results.append({
                    'backbone': backbone_name,
                    'feature_dim': feat_dim,
                    'method': method,
                    'components': n_comp,
                    'accuracy': result['accuracy'],
                    'reduction_sec': result['reduction_sec'],
                    'train_sec': result['train_sec']
                })
                
                print(f"  {method.upper():3s} @ {n_comp:2d} comp: {result['accuracy']:.4f}")
    
    backbone_df = pd.DataFrame(backbone_results)
    backbone_df.to_csv(results_dir / "backbone_comparison.csv", index=False)
    
    # ========================================================================
    # SECTION 3: Modern Methods Comparison (using ResNet-18)
    # ========================================================================
    print("\n" + "=" * 70)
    print("SECTION 3: Modern Methods Comparison")
    print("=" * 70)
    
    resnet_data = backbone_data['resnet18']
    X_train = resnet_data['X_train']
    y_train = resnet_data['y_train']
    X_test = resnet_data['X_test']
    y_test = resnet_data['y_test']
    
    modern_results = []
    
    for n_comp in [20, 40, 80, 99]:
        print(f"\nComponents: {n_comp}")
        
        # LDA + Logistic (baseline)
        result_lda = run_reduction_experiment(
            X_train, y_train, X_test, y_test, 'lda', n_comp
        )
        modern_results.append({
            'method': 'LDA + Logistic',
            'components': n_comp,
            'accuracy': result_lda['accuracy'],
            'time_sec': result_lda['reduction_sec'] + result_lda['train_sec']
        })
        print(f"  LDA + Logistic: {result_lda['accuracy']:.4f}")
        
        # PCA + Logistic
        result_pca = run_reduction_experiment(
            X_train, y_train, X_test, y_test, 'pca', n_comp
        )
        modern_results.append({
            'method': 'PCA + Logistic',
            'components': n_comp,
            'accuracy': result_pca['accuracy'],
            'time_sec': result_pca['reduction_sec'] + result_pca['train_sec']
        })
        print(f"  PCA + Logistic: {result_pca['accuracy']:.4f}")
        
        # LDA + KNN (SupCon proxy)
        result_supcon = run_supervised_contrastive_proxy(
            X_train, y_train, X_test, y_test, n_comp
        )
        modern_results.append({
            'method': 'LDA + KNN (SupCon proxy)',
            'components': n_comp,
            'accuracy': result_supcon['accuracy'],
            'time_sec': result_supcon['reduction_sec'] + result_supcon['eval_sec']
        })
        print(f"  LDA + KNN (SupCon proxy): {result_supcon['accuracy']:.4f}")
        
        # Metric Learning (NCA)
        print(f"  Training NCA metric learning...")
        result_nca = run_metric_learning_experiment(
            X_train, y_train, X_test, y_test, n_comp
        )
        modern_results.append({
            'method': 'NCA Metric Learning',
            'components': n_comp,
            'accuracy': result_nca['accuracy'],
            'time_sec': result_nca['train_sec'] + result_nca['eval_sec']
        })
        print(f"  NCA Metric Learning: {result_nca['accuracy']:.4f}")
    
    modern_df = pd.DataFrame(modern_results)
    modern_df.to_csv(results_dir / "modern_methods_comparison.csv", index=False)
    
    # ========================================================================
    # SECTION 4: Class-wise Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("SECTION 4: Class-wise LDA Improvement Analysis")
    print("=" * 70)
    
    # Run at 99 components for detailed analysis
    result_lda_99 = run_reduction_experiment(
        X_train, y_train, X_test, y_test, 'lda', 99
    )
    result_pca_99 = run_reduction_experiment(
        X_train, y_train, X_test, y_test, 'pca', 99
    )
    
    class_analysis = analyze_lda_class_improvement(
        result_lda_99, result_pca_99, y_test
    )
    class_analysis.to_csv(results_dir / "class_analysis.csv", index=False)
    
    # Top gainers and losers
    print("\nTop 10 classes that BENEFIT from LDA (vs PCA):")
    top_gainers = class_analysis.nlargest(10, 'improvement')
    for _, row in top_gainers.iterrows():
        print(f"  {row['class_name']:20s}: +{row['improvement']*100:5.1f}% "
              f"(LDA: {row['lda_acc']*100:.1f}%, PCA: {row['pca_acc']*100:.1f}%)")
    
    print("\nTop 10 classes where PCA is BETTER than LDA:")
    top_losers = class_analysis.nsmallest(10, 'improvement')
    for _, row in top_losers.iterrows():
        print(f"  {row['class_name']:20s}: {row['improvement']*100:5.1f}% "
              f"(LDA: {row['lda_acc']*100:.1f}%, PCA: {row['pca_acc']*100:.1f}%)")
    
    # Superclass analysis
    print("\nSuperclass-level LDA improvement:")
    superclass_improvement = class_analysis.groupby('superclass')['improvement'].mean()
    superclass_improvement = superclass_improvement.sort_values(ascending=False)
    for sc, imp in superclass_improvement.items():
        print(f"  {sc:30s}: {imp*100:+5.2f}%")
    
    # ========================================================================
    # SECTION 5: Edge Deployment Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("SECTION 5: Edge Deployment Analysis")
    print("=" * 70)
    
    edge_results = []
    
    for backbone_name, data in backbone_data.items():
        feat_dim = data['dim']
        
        for n_comp in [20, 40, 99]:
            metrics = compute_deployment_metrics(feat_dim, n_comp)
            
            # Get accuracy from backbone results
            acc_row = backbone_df[
                (backbone_df['backbone'] == backbone_name) & 
                (backbone_df['method'] == 'lda') &
                (backbone_df['components'] == n_comp)
            ]
            accuracy = acc_row['accuracy'].values[0] if len(acc_row) > 0 else 0
            
            edge_results.append({
                'backbone': backbone_name,
                'feature_dim': feat_dim,
                'lda_components': n_comp,
                'accuracy': accuracy,
                'original_memory_mb': metrics['original_memory_mb'],
                'reduced_memory_mb': metrics['reduced_memory_mb'],
                'memory_reduction_x': metrics['memory_reduction'],
                'model_size_mb': metrics['total_model_mb'],
                'inference_flops': metrics['total_flops'],
                'flops_reduction_x': metrics['flops_reduction']
            })
    
    edge_df = pd.DataFrame(edge_results)
    edge_df.to_csv(results_dir / "edge_deployment.csv", index=False)
    
    print("\nEdge Deployment Summary (LDA @ 40 components):")
    print("-" * 80)
    print(f"{'Backbone':<20} {'Feat Dim':<10} {'Acc %':<8} {'Mem Red':<10} {'Model MB':<10} {'FLOPs':<12}")
    print("-" * 80)
    
    for backbone_name in BACKBONES.keys():
        row = edge_df[(edge_df['backbone'] == backbone_name) & 
                      (edge_df['lda_components'] == 40)].iloc[0]
        print(f"{backbone_name:<20} {int(row['feature_dim']):<10} "
              f"{row['accuracy']*100:<8.2f} {row['memory_reduction_x']:<10.1f}x "
              f"{row['model_size_mb']:<10.3f} {int(row['inference_flops']):<12}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXTENDED STUDY SUMMARY")
    print("=" * 70)
    
    # Key finding 1: LDA advantage across backbones
    print("\n1. LDA Advantage Across Backbones (at 99 components):")
    print("-" * 60)
    for backbone_name in BACKBONES.keys():
        lda_acc = backbone_df[
            (backbone_df['backbone'] == backbone_name) & 
            (backbone_df['method'] == 'lda') &
            (backbone_df['components'] == 99)
        ]['accuracy'].values[0]
        pca_acc = backbone_df[
            (backbone_df['backbone'] == backbone_name) & 
            (backbone_df['method'] == 'pca') &
            (backbone_df['components'] == 99)
        ]['accuracy'].values[0]
        advantage = (lda_acc - pca_acc) * 100
        print(f"  {backbone_name:<20}: LDA={lda_acc*100:.2f}%, PCA={pca_acc*100:.2f}%, "
              f"Δ={advantage:+.2f}%")
    
    # Key finding 2: Modern methods
    print("\n2. Modern Methods Comparison (at 99 components):")
    print("-" * 60)
    for method in modern_df['method'].unique():
        row = modern_df[(modern_df['method'] == method) & 
                        (modern_df['components'] == 99)]
        if len(row) > 0:
            acc = row['accuracy'].values[0]
            time_s = row['time_sec'].values[0]
            print(f"  {method:<25}: {acc*100:.2f}% (time: {time_s:.2f}s)")
    
    # Key finding 3: Best classes for LDA
    print("\n3. Classes with Largest LDA Improvement:")
    print("-" * 60)
    top5 = class_analysis.nlargest(5, 'improvement')
    for _, row in top5.iterrows():
        print(f"  {row['class_name']:<20} ({row['superclass']:<25}): "
              f"+{row['improvement']*100:.1f}%")
    
    print("\n" + "=" * 70)
    print("Results saved to:")
    print(f"  - {results_dir / 'backbone_comparison.csv'}")
    print(f"  - {results_dir / 'modern_methods_comparison.csv'}")
    print(f"  - {results_dir / 'class_analysis.csv'}")
    print(f"  - {results_dir / 'edge_deployment.csv'}")
    print("=" * 70)
    
    return {
        'backbone_df': backbone_df,
        'modern_df': modern_df,
        'class_analysis': class_analysis,
        'edge_df': edge_df
    }


if __name__ == "__main__":
    results = run_extended_study()
