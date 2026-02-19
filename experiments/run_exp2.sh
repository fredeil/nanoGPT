#!/bin/bash
# Experiment 2 - Width sweep (n_embd)
# Fixed: n_layer=6, n_head=8, dropout=0.0
# Vary: n_embd ∈ {128, 192, 256, 384}

for WIDTH in 128 192 256 384; do
    echo "=========================================="
    echo "Training with n_embd=${WIDTH}"
    echo "=========================================="

    OUT_DIR="out-exp2-embd${WIDTH}"
    mkdir -p "$OUT_DIR"

    PYTHONUNBUFFERED=1 uv run train.py config/exp2_width.py \
        --out_dir="$OUT_DIR" \
        --n_embd=$WIDTH \
        --wandb_run_name="exp2-embd${WIDTH}" \
        2>&1 | tee "${OUT_DIR}/log.txt"

    echo ""
    echo "Sampling from n_embd=${WIDTH} model..."
    echo ""

    PYTHONUNBUFFERED=1 uv run sample.py \
        --out_dir="$OUT_DIR" \
        --device=mps \
        --dtype=float16 \
        --max_new_tokens=1000 \
        --num_samples=5 \
        2>&1 | tee "${OUT_DIR}/samples.txt"

    echo ""
    echo "Done with n_embd=${WIDTH}"
    echo ""
done

echo "Experiment 2 complete!"
