#!/bin/bash
# Experiment 1 - Depth sweep (n_layer)
# Fixed: n_head=8, n_embd=256, dropout=0.0
# Vary: n_layer ∈ {2, 4, 8, 12}

for LAYERS in 2 4 8 12; do
    echo "=========================================="
    echo "Training with n_layer=${LAYERS}"
    echo "=========================================="

    OUT_DIR="out-exp1-layer${LAYERS}"
    mkdir -p "$OUT_DIR"

    PYTHONUNBUFFERED=1 uv run train.py config/exp1_depth.py \
        --out_dir="$OUT_DIR" \
        --n_layer=$LAYERS \
        --wandb_run_name="exp1-layer${LAYERS}" \
        2>&1 | tee "${OUT_DIR}/log.txt"

    echo ""
    echo "Sampling from n_layer=${LAYERS} model..."
    echo ""

    PYTHONUNBUFFERED=1 uv run sample.py \
        --out_dir="$OUT_DIR" \
        --device=mps \
        --dtype=float16 \
        --max_new_tokens=1000 \
        --num_samples=5 \
        2>&1 | tee "${OUT_DIR}/samples.txt"

    echo ""
    echo "Done with n_layer=${LAYERS}"
    echo ""
done

echo "Experiment 1 complete!"
