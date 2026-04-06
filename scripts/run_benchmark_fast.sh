#!/bin/bash
# run_benchmark_fast.sh — Parallel benchmark runner for GCP
# Strategy:
#   Phase A: LogReg only across all 18 configs (fast, ~2hrs)  
#   Phase B: MLP across all 18 configs (slower but optimized)
# Each backbone×dataset runs as a separate process for parallelism.

set -e
cd ~/lda-image-classification
source .venv/bin/activate

OUTPUT_DIR="results/extended_benchmark"
mkdir -p "$OUTPUT_DIR"

BACKBONES="resnet18 resnet50 mobilenetv3 efficientnet vit_b16 dinov2_vits14"
DATASETS="cifar100 tiny_imagenet cub200"

echo "════════════════════════════════════════════════════════════"
echo "  PHASE A: LogReg — all backbones × datasets (parallel)"
echo "  Started: $(date)"
echo "════════════════════════════════════════════════════════════"

# Phase A: LogReg — run up to 6 configs in parallel
PIDS_A=()
for bb in $BACKBONES; do
    for ds in $DATASETS; do
        LOG_FILE="$OUTPUT_DIR/${bb}_${ds}_logreg.log"
        echo "  Starting: $bb × $ds (LogReg) → $LOG_FILE"
        python experiments/run_extended_benchmark.py \
            --backbone $bb --dataset $ds --classifier LogReg \
            --seeds 5 --output-dir "$OUTPUT_DIR" \
            > "$LOG_FILE" 2>&1 &
        PIDS_A+=($!)
        
        # Limit to 6 parallel processes (to not overwhelm memory)
        if [ ${#PIDS_A[@]} -ge 6 ]; then
            echo "  Waiting for batch of 6..."
            for pid in "${PIDS_A[@]}"; do
                wait $pid
            done
            PIDS_A=()
        fi
    done
done

# Wait for remaining
for pid in "${PIDS_A[@]}"; do
    wait $pid
done

echo ""
echo "  ✅ Phase A (LogReg) complete: $(date)"
echo "  Results:"
ls -la "$OUTPUT_DIR"/*.csv 2>/dev/null | wc -l
echo " CSV files saved"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  PHASE B: MLP — all backbones × datasets (parallel)"
echo "  Started: $(date)"
echo "════════════════════════════════════════════════════════════"

# Phase B: MLP — run up to 4 configs in parallel (MLP uses more memory)
PIDS_B=()
for bb in $BACKBONES; do
    for ds in $DATASETS; do
        LOG_FILE="$OUTPUT_DIR/${bb}_${ds}_mlp.log"
        echo "  Starting: $bb × $ds (MLP) → $LOG_FILE"
        python experiments/run_extended_benchmark.py \
            --backbone $bb --dataset $ds --classifier MLP \
            --seeds 5 --output-dir "$OUTPUT_DIR" \
            > "$LOG_FILE" 2>&1 &
        PIDS_B+=($!)
        
        # Limit to 4 parallel for MLP (more memory-intensive)
        if [ ${#PIDS_B[@]} -ge 4 ]; then
            echo "  Waiting for batch of 4..."
            for pid in "${PIDS_B[@]}"; do
                wait $pid
            done
            PIDS_B=()
        fi
    done
done

# Wait for remaining
for pid in "${PIDS_B[@]}"; do
    wait $pid
done

echo ""
echo "  ✅ Phase B (MLP) complete: $(date)"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Merging all results..."
echo "════════════════════════════════════════════════════════════"

# Merge all individual CSVs into one combined file
python3 -c "
import pandas as pd
import glob
import os

output_dir = '$OUTPUT_DIR'
csvs = sorted(glob.glob(os.path.join(output_dir, '*.csv')))
# Filter out the combined file itself
csvs = [f for f in csvs if 'all_results' not in os.path.basename(f)]

dfs = []
for f in csvs:
    try:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f'  Loaded: {os.path.basename(f)} ({len(df)} rows)')
    except Exception as e:
        print(f'  SKIP: {os.path.basename(f)}: {e}')

if dfs:
    combined = pd.concat(dfs, ignore_index=True)
    # Remove duplicates (keep last in case of reruns)
    combined = combined.drop_duplicates(
        subset=['backbone', 'dataset', 'method', 'classifier'],
        keep='last'
    )
    combined_path = os.path.join(output_dir, 'all_results.csv')
    combined.to_csv(combined_path, index=False)
    print(f'\\n  ✅ Combined: {len(combined)} rows → {combined_path}')
else:
    print('  ❌ No CSV files found!')
"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ✅ ALL DONE: $(date)"
echo "════════════════════════════════════════════════════════════"
