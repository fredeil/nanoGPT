#!/bin/bash
# Experiment 3 - Attention head sweep (n_head)
# Fixed: n_layer=6, n_embd=256, dropout=0.0
# Vary: n_head ∈ {4, 8, 16, 32}

for HEADS in 4 8 16 32; do
    echo "=========================================="
    echo "Training with n_head=${HEADS}"
    echo "=========================================="

    OUT_DIR="out-exp3-head${HEADS}"
    mkdir -p "$OUT_DIR"

    PYTHONUNBUFFERED=1 uv run train.py config/exp3_heads.py \
        --out_dir="$OUT_DIR" \
        --n_head=$HEADS \
        --wandb_run_name="exp3-head${HEADS}" \
        2>&1 | tee "${OUT_DIR}/log.txt"

    echo ""
    echo "Sampling from n_head=${HEADS} model..."
    echo ""

    PYTHONUNBUFFERED=1 uv run sample.py \
        --out_dir="$OUT_DIR" \
        --device=mps \
        --dtype=float16 \
        --max_new_tokens=1000 \
        --num_samples=5 \
        2>&1 | tee "${OUT_DIR}/samples.txt"

    echo ""
    echo "Done with n_head=${HEADS}"
    echo ""
done

echo "Experiment 3 complete!"
