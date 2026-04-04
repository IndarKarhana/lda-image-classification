#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Phase 3 GCP Runner — Automated VM lifecycle
# ═══════════════════════════════════════════════════════════════════════
# Creates a VM, uploads code+features, runs Phase 3, downloads results,
# and DELETES the VM — all in one script.
#
# Usage:
#   bash scripts/run_phase3_gcp.sh
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ──
PROJECT="gemeni-codebridge"
VM_NAME="lda-phase3"
ZONE="us-central1-a"
MACHINE_TYPE="n2-standard-16"      # 16 vCPUs, 64GB RAM (no GPU needed — CPU-only experiments)
DISK_SIZE="100GB"
IMAGE_FAMILY="debian-12"
IMAGE_PROJECT="debian-cloud"

LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="/home/$(gcloud config get-value account 2>/dev/null | cut -d@ -f1)/lda-project"
REMOTE_USER="$(gcloud config get-value account 2>/dev/null | cut -d@ -f1)"

echo "═══════════════════════════════════════════════════════════════"
echo "  Phase 3 GCP Runner"
echo "  VM: $VM_NAME ($MACHINE_TYPE)"
echo "  Local: $LOCAL_DIR"
echo "  Remote: $REMOTE_DIR"
echo "═══════════════════════════════════════════════════════════════"

# Track start time for cost estimation
START_TIME=$(date +%s)

# ── Step 1: Create VM ──
echo ""
echo "═══ Step 1: Creating GCP VM ═══"
gcloud compute instances create "$VM_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --boot-disk-size="$DISK_SIZE" \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --scopes="default" \
    --metadata="startup-script=#!/bin/bash
apt-get update -qq && apt-get install -y -qq python3-venv python3-pip > /dev/null 2>&1"

echo "VM created. Waiting 30s for boot..."
sleep 30

# Wait for SSH to be ready
echo "Waiting for SSH..."
for i in $(seq 1 12); do
    if gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT" --command="echo ready" 2>/dev/null; then
        break
    fi
    echo "  Attempt $i/12..."
    sleep 10
done

# ── Step 2: Upload code + features ──
echo ""
echo "═══ Step 2: Uploading code + features ═══"

# Create remote directory structure
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT" --command="
mkdir -p $REMOTE_DIR/{data,features/{saved,tiny_imagenet},reduction,models,experiments,results/{academic_benchmark,phase3},scripts}
"

# Upload Python files (exclude data, venv, large files)
echo "  Uploading Python source..."
for dir in data features reduction models experiments; do
    gcloud compute scp --recurse --zone="$ZONE" --project="$PROJECT" \
        "$LOCAL_DIR/$dir/"*.py "$VM_NAME:$REMOTE_DIR/$dir/" 2>/dev/null || true
done

# Upload __init__.py files
for dir in data features reduction models experiments; do
    gcloud compute scp --zone="$ZONE" --project="$PROJECT" \
        "$LOCAL_DIR/$dir/__init__.py" "$VM_NAME:$REMOTE_DIR/$dir/" 2>/dev/null || true
done

# Upload cached features (CIFAR-100)
echo "  Uploading cached features..."
for f in "$LOCAL_DIR"/features/saved/*.npz; do
    [ -f "$f" ] && gcloud compute scp --zone="$ZONE" --project="$PROJECT" \
        "$f" "$VM_NAME:$REMOTE_DIR/features/saved/"
done

# Upload Tiny ImageNet features if they exist locally
if [ -d "$LOCAL_DIR/features/tiny_imagenet" ]; then
    for f in "$LOCAL_DIR"/features/tiny_imagenet/*.npz; do
        [ -f "$f" ] && gcloud compute scp --zone="$ZONE" --project="$PROJECT" \
            "$f" "$VM_NAME:$REMOTE_DIR/features/tiny_imagenet/"
    done
fi

# Upload Phase 2 results (for cost analysis)
echo "  Uploading Phase 2 results..."
gcloud compute scp --zone="$ZONE" --project="$PROJECT" \
    "$LOCAL_DIR/results/academic_benchmark/all_benchmarks.csv" \
    "$VM_NAME:$REMOTE_DIR/results/academic_benchmark/" 2>/dev/null || true

echo "  Upload complete."

# ── Step 3: Install dependencies ──
echo ""
echo "═══ Step 3: Installing Python dependencies ═══"
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT" --command="
cd $REMOTE_DIR
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install torch torchvision numpy pandas scikit-learn scipy tqdm -q
echo 'Dependencies installed.'
python3 -c 'import sklearn; import torch; print(f\"sklearn={sklearn.__version__}, torch={torch.__version__}\")'
"

# ── Step 4: Extract missing features (Tiny ImageNet if needed) ──
echo ""
echo "═══ Step 4: Check / extract features ═══"
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT" --command="
cd $REMOTE_DIR
source .venv/bin/activate

# Check what we have
echo 'CIFAR-100 features:'
ls -lh features/saved/*.npz 2>/dev/null || echo '  (none)'
echo 'Tiny ImageNet features:'
ls -lh features/tiny_imagenet/*.npz 2>/dev/null || echo '  (none)'

# Extract any missing features
python3 -c '
import os, sys
sys.path.insert(0, \".\")
from features.extract_features_multi import BACKBONES, get_or_extract_cifar100, get_or_extract_tiny_imagenet

for bb in BACKBONES:
    # CIFAR-100
    path_c = f\"features/saved/{bb}_cifar100.npz\"
    if not os.path.exists(path_c):
        print(f\"  Extracting {bb} CIFAR-100...\")
        get_or_extract_cifar100(bb)
    else:
        print(f\"  ✓ {bb} CIFAR-100 cached\")

    # Tiny ImageNet
    path_t = f\"features/tiny_imagenet/{bb}_features.npz\"
    if not os.path.exists(path_t):
        print(f\"  Extracting {bb} Tiny ImageNet...\")
        get_or_extract_tiny_imagenet(bb)
    else:
        print(f\"  ✓ {bb} Tiny ImageNet cached\")

print(\"All features ready.\")
'
"

# ── Step 5: Run Phase 3 experiments ──
echo ""
echo "═══ Step 5: Running Phase 3 experiments ═══"
echo "  Started at: $(date)"

gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT" --command="
cd $REMOTE_DIR
source .venv/bin/activate

echo 'Running Phase 3 — all experiments...'
python3 experiments/run_phase3_experiments.py --all 2>&1 | tee phase3_run.log

echo ''
echo 'Results:'
ls -lh results/phase3/
" 2>&1 | tee /tmp/phase3_gcp_output.log

echo ""
echo "  Finished at: $(date)"

# ── Step 6: Download results ──
echo ""
echo "═══ Step 6: Downloading results ═══"

mkdir -p "$LOCAL_DIR/results/phase3"

# Download all Phase 3 CSVs
gcloud compute scp --recurse --zone="$ZONE" --project="$PROJECT" \
    "$VM_NAME:$REMOTE_DIR/results/phase3/" \
    "$LOCAL_DIR/results/phase3/"

# Download run log
gcloud compute scp --zone="$ZONE" --project="$PROJECT" \
    "$VM_NAME:$REMOTE_DIR/phase3_run.log" \
    "$LOCAL_DIR/results/phase3/phase3_run.log" 2>/dev/null || true

# Download any extracted Tiny ImageNet features we didn't have
if [ ! -d "$LOCAL_DIR/features/tiny_imagenet" ] || [ -z "$(ls "$LOCAL_DIR/features/tiny_imagenet/"*.npz 2>/dev/null)" ]; then
    echo "  Downloading Tiny ImageNet features..."
    mkdir -p "$LOCAL_DIR/features/tiny_imagenet"
    gcloud compute scp --recurse --zone="$ZONE" --project="$PROJECT" \
        "$VM_NAME:$REMOTE_DIR/features/tiny_imagenet/" \
        "$LOCAL_DIR/features/tiny_imagenet/" 2>/dev/null || true
fi

# Download EfficientNet features if missing locally
if [ ! -f "$LOCAL_DIR/features/saved/efficientnet_cifar100.npz" ]; then
    echo "  Downloading EfficientNet CIFAR-100 features..."
    gcloud compute scp --zone="$ZONE" --project="$PROJECT" \
        "$VM_NAME:$REMOTE_DIR/features/saved/efficientnet_cifar100.npz" \
        "$LOCAL_DIR/features/saved/" 2>/dev/null || true
fi

echo "  Downloaded results:"
ls -lh "$LOCAL_DIR/results/phase3/"

# ── Step 7: DELETE VM ──
echo ""
echo "═══ Step 7: Deleting GCP VM ═══"
gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --project="$PROJECT" --quiet

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✅ PHASE 3 COMPLETE"
echo "  Total wall time: ${ELAPSED} minutes"
echo "  VM $VM_NAME deleted — no ongoing charges"
echo "  Results in: $LOCAL_DIR/results/phase3/"
echo "═══════════════════════════════════════════════════════════════"
