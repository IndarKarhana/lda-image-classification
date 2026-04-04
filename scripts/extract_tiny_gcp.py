"""Extract Tiny ImageNet features for all 4 backbones on GCP VM."""
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import sys
import os
import time

# Ensure we're in the project root
project_root = os.path.expanduser('~/lda-project')
os.chdir(project_root)
sys.path.insert(0, project_root)

from features.extract_features_multi import get_or_extract_tiny_imagenet

backbones = ['resnet18', 'resnet50', 'mobilenetv3', 'efficientnet']

for bb in backbones:
    print()
    print('=' * 50)
    print('Extracting: {} x tiny_imagenet'.format(bb))
    print('=' * 50)
    t0 = time.perf_counter()
    X_tr, y_tr, X_te, y_te, dim = get_or_extract_tiny_imagenet(bb, batch_size=64)
    elapsed = time.perf_counter() - t0
    print('Done in {:.1f}s - train={}, test={}'.format(elapsed, X_tr.shape, X_te.shape))

print()
print('All Tiny ImageNet features extracted!')
