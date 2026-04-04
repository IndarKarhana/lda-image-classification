"""Helper script to extract all missing features on GCP VM."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.extract_features_multi import BACKBONES, get_or_extract_cifar100, get_or_extract_tiny_imagenet

for bb in BACKBONES:
    path_c = f"features/saved/{bb}_cifar100.npz"
    if not os.path.exists(path_c):
        print(f"  Extracting {bb} CIFAR-100...")
        get_or_extract_cifar100(bb)
    else:
        print(f"  ✓ {bb} CIFAR-100 cached")

for bb in BACKBONES:
    path_t = f"features/tiny_imagenet/{bb}_features.npz"
    if not os.path.exists(path_t):
        print(f"  Extracting {bb} Tiny ImageNet...")
        get_or_extract_tiny_imagenet(bb)
    else:
        print(f"  ✓ {bb} Tiny ImageNet cached")

print("All features ready.")
