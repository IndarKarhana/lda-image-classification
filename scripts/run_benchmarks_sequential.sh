#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Sequential Academic Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════
# Runs benchmarks ONE AT A TIME for:
#   1. Accurate timing measurements (no CPU contention)
#   2. Reliable comparisons across methods
#
# Usage:
#   bash scripts/run_benchmarks_sequential.sh                # default: 3 cached backbones
#   bash scripts/run_benchmarks_sequential.sh cifar100       # CIFAR-100 only
#   bash scripts/run_benchmarks_sequential.sh tiny_imagenet  # Tiny ImageNet only
#   bash scripts/run_benchmarks_sequential.sh all            # both datasets
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${1:-cifar100}"
RESULTS_DIR="results/academic_benchmark"
mkdir -p "$RESULTS_DIR"

# Determine which backbones have cached features
CIFAR_BACKBONES=()
for bb in resnet18 resnet50 mobilenetv3 efficientnet; do
    if [[ -f "features/saved/${bb}_cifar100.npz" ]]; then
        CIFAR_BACKBONES+=("$bb")
    fi
done

TINY_BACKBONES=()
for bb in resnet18 resnet50 mobilenetv3 efficientnet; do
    if [[ -f "features/tiny_imagenet/${bb}_features.npz" ]]; then
        TINY_BACKBONES+=("$bb")
    fi
done

echo "═══════════════════════════════════════════════════════════════"
echo "  Sequential Academic Benchmark Runner"
echo "  Dataset: $DATASET"
echo "  CIFAR-100 backbones with cached features: ${CIFAR_BACKBONES[*]:-none}"
echo "  Tiny ImageNet backbones with cached features: ${TINY_BACKBONES[*]:-none}"
echo "═══════════════════════════════════════════════════════════════"
echo ""

run_one() {
    local bb="$1"
    local ds="$2"
    local logfile="$RESULTS_DIR/${bb}_${ds}_log.txt"
    
    echo "━━━ Starting: $bb × $ds ━━━"
    echo "    Log: $logfile"
    echo "    Time: $(date '+%H:%M:%S')"
    
    python experiments/run_academic_benchmark.py \
        --backbone "$bb" --dataset "$ds" \
        2>&1 | tee "$logfile"
    
    echo "    ✅ Completed: $bb × $ds at $(date '+%H:%M:%S')"
    echo ""
}

if [[ "$DATASET" == "cifar100" || "$DATASET" == "all" ]]; then
    echo "═══ CIFAR-100 Benchmarks ═══"
    for bb in "${CIFAR_BACKBONES[@]}"; do
        run_one "$bb" "cifar100"
    done
fi

if [[ "$DATASET" == "tiny_imagenet" || "$DATASET" == "all" ]]; then
    echo "═══ Tiny ImageNet Benchmarks ═══"
    if [[ ${#TINY_BACKBONES[@]} -eq 0 ]]; then
        echo "  ⚠️  No Tiny ImageNet features cached. Extract first with:"
        echo "     python features/extract_features_multi.py --datasets tiny_imagenet"
    else
        for bb in "${TINY_BACKBONES[@]}"; do
            run_one "$bb" "tiny_imagenet"
        done
    fi
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  ✅ ALL BENCHMARKS COMPLETE"
echo "  Results in: $RESULTS_DIR/"
ls -la "$RESULTS_DIR"/*.csv 2>/dev/null || echo "  (no CSVs found)"
echo "═══════════════════════════════════════════════════════════════"
