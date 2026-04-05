#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# GCP Setup & Run Script — Extended Benchmark (v2)
# ═══════════════════════════════════════════════════════════════════════
#
# Creates a GCP VM with GPU, sets up the environment, extracts features
# (including ViT-B/16, DINOv2 ViT-S/14), and runs the extended benchmark
# with MLP classifier comparison across CIFAR-100, Tiny ImageNet, CUB-200.
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - Project set: gcloud config set project <PROJECT_ID>
#   - GPU quota available in zone
#
# Usage:
#   bash scripts/gcp_setup_v2.sh create       # Create VM + setup
#   bash scripts/gcp_setup_v2.sh upload        # Upload code + cached features
#   bash scripts/gcp_setup_v2.sh extract       # Extract features (all backbones × datasets)
#   bash scripts/gcp_setup_v2.sh run           # Run extended benchmark
#   bash scripts/gcp_setup_v2.sh download      # Download results + features
#   bash scripts/gcp_setup_v2.sh cleanup       # Delete VM
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ──
VM_NAME="lda-extended"
ZONE="us-central1-a"
MACHINE_TYPE="n2-standard-16"      # 16 vCPUs, 64GB RAM (need RAM for DINOv2 + large features)
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
DISK_SIZE="200GB"                  # CUB-200 ~1.1GB, ImageNet features are large
IMAGE_FAMILY="pytorch-latest-gpu"
IMAGE_PROJECT="deeplearning-platform-release"

REPO_URL="https://github.com/IndarKarhana/lda-image-classification.git"
REMOTE_DIR="/home/\$USER/lda-image-classification"

# All 6 backbones
CNN_BACKBONES="resnet18 resnet50 mobilenetv3 efficientnet"
VIT_BACKBONES="vit_b16 dinov2_vits14"
ALL_BACKBONES="$CNN_BACKBONES $VIT_BACKBONES"

# All 3 datasets
ALL_DATASETS="cifar100 tiny_imagenet cub200"

# ═══════════════════════════════════════════════════════════════════════
case "${1:-help}" in

create)
    echo "═══ Creating GCP VM: $VM_NAME ═══"
    
    gcloud compute instances create "$VM_NAME" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
        --boot-disk-size="$DISK_SIZE" \
        --image-family="$IMAGE_FAMILY" \
        --image-project="$IMAGE_PROJECT" \
        --maintenance-policy=TERMINATE \
        --metadata="install-nvidia-driver=True" \
        --scopes="default,storage-rw"
    
    echo ""
    echo "VM created. Waiting 90s for boot + GPU driver..."
    sleep 90
    
    echo "═══ Setting up environment ═══"
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
        # Clone repo
        git clone $REPO_URL ~/lda-image-classification || (cd ~/lda-image-classification && git pull)
        cd ~/lda-image-classification
        
        # Create venv
        python3 -m venv .venv
        source .venv/bin/activate
        
        # Install dependencies
        pip install --upgrade pip
        pip install torch torchvision numpy pandas scikit-learn scipy matplotlib seaborn tqdm
        
        # Verify GPU
        python -c '
import torch
print(f\"CUDA available: {torch.cuda.is_available()}\")
if torch.cuda.is_available():
    print(f\"GPU: {torch.cuda.get_device_name(0)}\")
    print(f\"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB\")
'
        echo ''
        echo '═══ Setup complete ═══'
    "
    
    echo ""
    echo "✅ VM ready. SSH in with:"
    echo "   gcloud compute ssh $VM_NAME --zone=$ZONE"
    ;;

upload)
    echo "═══ Uploading local code + cached features to VM ═══"
    
    # Upload code
    gcloud compute scp --recurse --zone="$ZONE" \
        --exclude=".venv,data,__pycache__,*.pyc,.git,paper,archive,notebooks" \
        . "$VM_NAME:~/lda-image-classification/"
    
    # Upload existing cached features (saves re-extraction time)
    echo "Uploading cached CIFAR-100 features..."
    gcloud compute scp --recurse --zone="$ZONE" \
        features/saved/ "$VM_NAME:~/lda-image-classification/features/saved/" 2>/dev/null || true
    
    echo "Uploading cached Tiny ImageNet features..."
    gcloud compute scp --recurse --zone="$ZONE" \
        features/tiny_imagenet/ "$VM_NAME:~/lda-image-classification/features/tiny_imagenet/" 2>/dev/null || true
    
    echo "✅ Upload complete"
    ;;

extract)
    echo "═══ Extracting features on GCP (all backbones × all datasets) ═══"
    
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
        cd ~/lda-image-classification
        source .venv/bin/activate
        
        echo '═══ Phase 1: CNN backbones × existing datasets ═══'
        python features/extract_features_multi.py \
            --backbones $CNN_BACKBONES \
            --datasets cifar100 tiny_imagenet \
            --batch-size 128
        
        echo ''
        echo '═══ Phase 2: ViT/DINOv2 backbones × existing datasets ═══'
        python features/extract_features_multi.py \
            --backbones $VIT_BACKBONES \
            --datasets cifar100 tiny_imagenet \
            --batch-size 64
        
        echo ''
        echo '═══ Phase 3: All backbones × CUB-200-2011 ═══'
        # CUB-200 will be auto-downloaded on first use
        python features/extract_features_multi.py \
            --backbones $ALL_BACKBONES \
            --datasets cub200 \
            --batch-size 64
        
        echo ''
        echo '═══ Feature extraction complete ═══'
        echo 'CIFAR-100 features:'
        ls -lh features/saved/
        echo 'Tiny ImageNet features:'
        ls -lh features/tiny_imagenet/
        echo 'CUB-200 features:'
        ls -lh features/cub200/
    "
    ;;

run)
    echo "═══ Running Extended Benchmark on GCP ═══"
    echo "  6 backbones × 3 datasets × 10 methods × 2 classifiers × 5 seeds"
    echo ""
    
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
        cd ~/lda-image-classification
        source .venv/bin/activate
        
        echo '═══ Running full extended benchmark ═══'
        echo 'Start time:' \$(date)
        
        # Run everything
        python experiments/run_extended_benchmark.py \
            --backbone all \
            --dataset all \
            --seeds 5 \
            --output-dir results/extended_benchmark \
            2>&1 | tee results/extended_benchmark/run.log
        
        echo ''
        echo 'End time:' \$(date)
        echo '═══ ✅ Extended benchmark complete ═══'
        ls -la results/extended_benchmark/
    "
    ;;

run-incremental)
    # Run one backbone at a time (safer for long jobs)
    echo "═══ Running Extended Benchmark incrementally ═══"
    
    for BB in $ALL_BACKBONES; do
        for DS in $ALL_DATASETS; do
            echo "--- $BB × $DS ---"
            gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
                cd ~/lda-image-classification
                source .venv/bin/activate
                python experiments/run_extended_benchmark.py \
                    --backbone $BB \
                    --dataset $DS \
                    --seeds 5 \
                    --output-dir results/extended_benchmark \
                    2>&1 | tee -a results/extended_benchmark/incremental.log
            "
        done
    done
    
    echo "✅ All incremental runs complete"
    ;;

download)
    echo "═══ Downloading results + features from GCP ═══"
    
    # Download results
    mkdir -p results/extended_benchmark
    gcloud compute scp --recurse --zone="$ZONE" \
        "$VM_NAME:~/lda-image-classification/results/extended_benchmark/" \
        results/extended_benchmark/
    
    # Download MLP benchmark results too
    mkdir -p results/mlp_benchmark
    gcloud compute scp --recurse --zone="$ZONE" \
        "$VM_NAME:~/lda-image-classification/results/mlp_benchmark/" \
        results/mlp_benchmark/ 2>/dev/null || true
    
    # Download newly extracted features
    echo "Downloading ViT/DINOv2 features..."
    gcloud compute scp --recurse --zone="$ZONE" \
        "$VM_NAME:~/lda-image-classification/features/saved/" \
        features/saved/
    
    echo "Downloading Tiny ImageNet features..."
    gcloud compute scp --recurse --zone="$ZONE" \
        "$VM_NAME:~/lda-image-classification/features/tiny_imagenet/" \
        features/tiny_imagenet/
    
    echo "Downloading CUB-200 features..."
    mkdir -p features/cub200
    gcloud compute scp --recurse --zone="$ZONE" \
        "$VM_NAME:~/lda-image-classification/features/cub200/" \
        features/cub200/
    
    echo "✅ Download complete"
    echo ""
    echo "Results:"
    ls -la results/extended_benchmark/
    echo ""
    echo "Features:"
    ls -lh features/saved/ features/tiny_imagenet/ features/cub200/ 2>/dev/null || true
    ;;

status)
    echo "═══ Checking VM status ═══"
    gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --format="table(name,status,machineType,zone)"
    echo ""
    echo "═══ GPU utilization ═══"
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="nvidia-smi" 2>/dev/null || echo "Cannot connect to VM"
    ;;

cleanup)
    echo "═══ Deleting GCP VM: $VM_NAME ═══"
    echo "This will delete the VM and all its data."
    gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet
    echo "✅ VM deleted — no more charges"
    ;;

help|*)
    echo "Usage: bash scripts/gcp_setup_v2.sh {create|upload|extract|run|run-incremental|download|status|cleanup}"
    echo ""
    echo "  create          - Create GCP VM with T4 GPU + install deps"
    echo "  upload          - Upload code + cached features to VM"
    echo "  extract         - Extract features (all 6 backbones × 3 datasets)"
    echo "  run             - Run extended benchmark (all at once)"
    echo "  run-incremental - Run benchmark one config at a time (safer)"
    echo "  download        - Download results + features back to local"
    echo "  status          - Check VM status + GPU utilization"
    echo "  cleanup         - Delete the VM"
    echo ""
    echo "Typical workflow:"
    echo "  1. bash scripts/gcp_setup_v2.sh create"
    echo "  2. bash scripts/gcp_setup_v2.sh upload"
    echo "  3. bash scripts/gcp_setup_v2.sh extract"
    echo "  4. bash scripts/gcp_setup_v2.sh run"
    echo "  5. bash scripts/gcp_setup_v2.sh download"
    echo "  6. bash scripts/gcp_setup_v2.sh cleanup"
    echo ""
    echo "Backbones: $ALL_BACKBONES"
    echo "Datasets:  $ALL_DATASETS"
    ;;

esac
