#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# GCP Setup & Run Script — Academic Benchmark
# ═══════════════════════════════════════════════════════════════════════
#
# Creates a GCP VM with GPU, sets up the environment, extracts features,
# and runs the full academic benchmark.
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - Project set: gcloud config set project <PROJECT_ID>
#
# Usage:
#   bash scripts/gcp_setup.sh create      # Create VM + setup
#   bash scripts/gcp_setup.sh run         # Run experiments (SSH into VM)
#   bash scripts/gcp_setup.sh download    # Download results back
#   bash scripts/gcp_setup.sh cleanup     # Delete VM
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ──
VM_NAME="lda-benchmark"
ZONE="us-central1-a"
MACHINE_TYPE="n1-standard-8"      # 8 vCPUs, 30GB RAM
GPU_TYPE="nvidia-tesla-t4"         # T4: good price/perf for inference
GPU_COUNT=1
DISK_SIZE="100GB"
IMAGE_FAMILY="pytorch-latest-gpu"
IMAGE_PROJECT="deeplearning-platform-release"

REPO_URL="https://github.com/YOUR_USERNAME/lda-image-classification.git"  # UPDATE THIS
REMOTE_DIR="/home/$USER/lda-image-classification"

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
    echo "VM created. Waiting 60s for boot..."
    sleep 60
    
    echo "═══ Setting up environment ═══"
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
        # Clone repo
        git clone $REPO_URL $REMOTE_DIR || true
        cd $REMOTE_DIR
        
        # Create venv
        python3 -m venv .venv
        source .venv/bin/activate
        
        # Install dependencies
        pip install --upgrade pip
        pip install torch torchvision numpy pandas scikit-learn scipy matplotlib seaborn tqdm
        
        # Verify GPU
        python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\")'
        
        echo ''
        echo '═══ Setup complete ═══'
    "
    
    echo ""
    echo "✅ VM ready. SSH in with:"
    echo "   gcloud compute ssh $VM_NAME --zone=$ZONE"
    ;;

upload)
    echo "═══ Uploading local code + cached features to VM ═══"
    
    # Upload code (exclude data, venv, large files)
    gcloud compute scp --recurse --zone="$ZONE" \
        --exclude=".venv,data,__pycache__,*.pyc,.git" \
        . "$VM_NAME:$REMOTE_DIR/"
    
    # Upload cached features (saves re-extraction time)
    echo "Uploading cached CIFAR-100 features..."
    gcloud compute scp --recurse --zone="$ZONE" \
        features/saved/ "$VM_NAME:$REMOTE_DIR/features/saved/"
    
    echo "✅ Upload complete"
    ;;

run)
    echo "═══ Running full benchmark on GCP ═══"
    
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
        cd $REMOTE_DIR
        source .venv/bin/activate
        
        echo '═══ Step 1: Extract missing features ═══'
        
        # EfficientNet CIFAR-100 (fast on T4, impossible on MPS)
        python features/extract_features_multi.py --backbones efficientnet --datasets cifar100
        
        # All backbones × Tiny ImageNet
        python features/extract_features_multi.py --backbones resnet18 resnet50 mobilenetv3 efficientnet --datasets tiny_imagenet
        
        echo ''
        echo '═══ Step 2: Run benchmarks — CIFAR-100 ═══'
        
        # All 4 backbones × CIFAR-100
        for bb in resnet18 resnet50 mobilenetv3 efficientnet; do
            echo \"--- \$bb × cifar100 ---\"
            python experiments/run_academic_benchmark.py --backbone \$bb --dataset cifar100
        done
        
        echo ''
        echo '═══ Step 3: Run benchmarks — Tiny ImageNet ═══'
        
        # All 4 backbones × Tiny ImageNet
        for bb in resnet18 resnet50 mobilenetv3 efficientnet; do
            echo \"--- \$bb × tiny_imagenet ---\"
            python experiments/run_academic_benchmark.py --backbone \$bb --dataset tiny_imagenet
        done
        
        echo ''
        echo '═══ ✅ All benchmarks complete ═══'
        ls -la results/academic_benchmark/
    "
    ;;

download)
    echo "═══ Downloading results from GCP ═══"
    
    # Download results CSVs
    gcloud compute scp --recurse --zone="$ZONE" \
        "$VM_NAME:$REMOTE_DIR/results/academic_benchmark/" \
        results/academic_benchmark/
    
    # Download extracted features (so we don't need to re-extract)
    gcloud compute scp --recurse --zone="$ZONE" \
        "$VM_NAME:$REMOTE_DIR/features/saved/" \
        features/saved/
    gcloud compute scp --recurse --zone="$ZONE" \
        "$VM_NAME:$REMOTE_DIR/features/tiny_imagenet/" \
        features/tiny_imagenet/
    
    echo "✅ Download complete"
    echo "Results:"
    ls -la results/academic_benchmark/
    ;;

cleanup)
    echo "═══ Deleting GCP VM ═══"
    gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet
    echo "✅ VM deleted"
    ;;

help|*)
    echo "Usage: bash scripts/gcp_setup.sh {create|upload|run|download|cleanup}"
    echo ""
    echo "  create    - Create GCP VM with T4 GPU + install deps"
    echo "  upload    - Upload code + cached features to VM"
    echo "  run       - Run full benchmark (all backbones × datasets)"
    echo "  download  - Download results + features back to local"
    echo "  cleanup   - Delete the VM"
    ;;

esac
