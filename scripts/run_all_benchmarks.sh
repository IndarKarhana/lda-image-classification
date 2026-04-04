#!/bin/bash
# ================================================================
# Master Benchmark Runner for GCP VM
# Runs all backbone × dataset combinations SEQUENTIALLY
# for accurate timing measurements.
# ================================================================
set -e

cd ~/lda-project
source .venv/bin/activate
export PYTHONUNBUFFERED=1

LOGDIR="/tmp/benchmark_logs"
mkdir -p "$LOGDIR"
mkdir -p results/academic_benchmark

echo "═══════════════════════════════════════════════════════════"
echo "  ACADEMIC BENCHMARK — FULL SWEEP"
echo "  $(date)"
echo "  $(nproc) CPUs, $(free -h | awk '/Mem:/{print $2}') RAM"
echo "═══════════════════════════════════════════════════════════"

# ── CIFAR-100 benchmarks (features already cached) ──
for BB in resnet18 resnet50 mobilenetv3 efficientnet; do
    echo ""
    echo "════════════════════════════════════════"
    echo "  Starting: ${BB} × CIFAR-100"
    echo "  $(date)"
    echo "════════════════════════════════════════"
    
    python3 experiments/run_academic_benchmark.py \
        --backbone "$BB" --dataset cifar100 \
        2>&1 | tee "$LOGDIR/${BB}_cifar100.log"
    
    echo "  ✅ ${BB} × CIFAR-100 done at $(date)"
done

# ── Tiny ImageNet benchmarks (features must be cached first) ──
for BB in resnet18 resnet50 mobilenetv3 efficientnet; do
    FEAT="features/tiny_imagenet/${BB}_features.npz"
    if [ ! -f "$FEAT" ]; then
        echo "  ⚠️  Skipping ${BB} × Tiny ImageNet — features not found: $FEAT"
        continue
    fi
    
    echo ""
    echo "════════════════════════════════════════"
    echo "  Starting: ${BB} × Tiny ImageNet"
    echo "  $(date)"
    echo "════════════════════════════════════════"
    
    python3 experiments/run_academic_benchmark.py \
        --backbone "$BB" --dataset tiny_imagenet \
        2>&1 | tee "$LOGDIR/${BB}_tiny_imagenet.log"
    
    echo "  ✅ ${BB} × Tiny ImageNet done at $(date)"
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ALL BENCHMARKS COMPLETE — $(date)"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Results:"
ls -lh results/academic_benchmark/
