#!/bin/bash
# Start the extended benchmark on GCP VM
gcloud compute ssh lda-extended --zone=us-central1-a --command="cd ~/lda-image-classification && mkdir -p results/extended_benchmark && nohup bash -c 'source .venv/bin/activate && python experiments/run_extended_benchmark.py --backbone all --dataset all --seeds 5 --output-dir results/extended_benchmark' > ~/benchmark.log 2>&1 &"
echo "Benchmark started on VM. Check with: gcloud compute ssh lda-extended --zone=us-central1-a --command='tail -20 ~/benchmark.log'"
