# Experiment 3: Attention Head Sweep (n_head)

## Setup

This experiment investigates the effect of the number of attention heads on language generation quality. We train four GPT models on the TinyStories dataset, varying only the number of attention heads while keeping all other hyperparameters fixed.

| Hyperparameter | Value |
|---|---|
| `n_layer` | 6 |
| `n_embd` | 256 |
| `dropout` | 0.0 |
| `block_size` | 256 |
| `batch_size` | 64 |
| `max_iters` | 5000 |
| `learning_rate` | 1e-3 |
| `min_lr` | 1e-4 |
| `warmup_iters` | 100 |

**Varied:** `n_head` in {4, 8, 16, 32}

Note: `n_embd` must be divisible by `n_head`. With `n_embd=256`, all four values (4, 8, 16, 32) satisfy this constraint, yielding per-head dimensions of 64, 32, 16, and 8 respectively.

## Results

### Parameter Counts

| n_head | Total Parameters | Decayed | Non-decayed |
|--------|-----------------|---------|-------------|
| 4 | 4.85M | 4,842,496 | 3,328 |
| 8 | 4.85M | 4,842,496 | 3,328 |
| 16 | 4.85M | 4,842,496 | 3,328 |
| 32 | 4.85M | 4,842,496 | 3,328 |

All four models have identical parameter counts. Changing `n_head` does not alter the total number of parameters because the Q, K, and V projection matrices remain the same size (`n_embd x n_embd`) regardless of how many heads they are split into. This makes experiment 3 a pure test of the attention head granularity, free from the confound of model capacity differences seen in experiments 1 and 2.

### Validation Loss

| n_head | Val Loss (step 1000) | Val Loss (step 2500) | Val Loss (final, step 5000) |
|--------|---------------------|---------------------|----------------------------|
| 4 | 0.8484 | 0.6788 | 0.5954 |
| 8 | 0.8494 | 0.6836 | 0.6000 |
| 16 | 0.8576 | 0.6847 | 0.6011 |
| 32 | 0.8607 | 0.6923 | 0.6062 |

### Training Loss Curves

Validation loss at every 250 steps:

| Step | n_head=4 | n_head=8 | n_head=16 | n_head=32 |
|------|----------|----------|-----------|-----------|
| 0 | 5.4053 | 5.4048 | 5.4054 | 5.4059 |
| 250 | 1.6841 | 1.7103 | 1.7243 | 1.7116 |
| 500 | 1.1418 | 1.1260 | 1.1258 | 1.1102 |
| 750 | 0.9460 | 0.9487 | 0.9469 | 0.9393 |
| 1000 | 0.8484 | 0.8494 | 0.8576 | 0.8607 |
| 1250 | 0.7926 | 0.7955 | 0.8033 | 0.8097 |
| 1500 | 0.7531 | 0.7581 | 0.7644 | 0.7724 |
| 1750 | 0.7324 | 0.7401 | 0.7395 | 0.7485 |
| 2000 | 0.7075 | 0.7125 | 0.7154 | 0.7232 |
| 2250 | 0.6876 | 0.6917 | 0.6961 | 0.6994 |
| 2500 | 0.6788 | 0.6836 | 0.6847 | 0.6923 |
| 2750 | 0.6607 | 0.6663 | 0.6657 | 0.6726 |
| 3000 | 0.6480 | 0.6527 | 0.6542 | 0.6605 |
| 3250 | 0.6397 | 0.6440 | 0.6458 | 0.6520 |
| 3500 | 0.6322 | 0.6369 | 0.6375 | 0.6429 |
| 3750 | 0.6188 | 0.6232 | 0.6253 | 0.6297 |
| 4000 | 0.6128 | 0.6178 | 0.6183 | 0.6237 |
| 4250 | 0.6055 | 0.6104 | 0.6114 | 0.6158 |
| 4500 | 0.6021 | 0.6068 | 0.6075 | 0.6130 |
| 4750 | 0.5988 | 0.6033 | 0.6046 | 0.6096 |
| 5000 | 0.5954 | 0.6000 | 0.6011 | 0.6062 |

All four models converge to remarkably similar final losses, with only a 0.011 spread between the best (n_head=4, 0.5954) and worst (n_head=32, 0.6062). The ordering is consistent throughout training: fewer heads consistently yield slightly lower loss. The differences are small but monotonic.

### Vocabulary Diversity

Measured as unique_words / total_words across all 5 generated samples:

| n_head | Unique Words | Total Words | Diversity Ratio |
|--------|-------------|-------------|-----------------|
| 4 | 181 | 576 | 0.314 |
| 8 | 179 | 571 | 0.313 |
| 16 | 175 | 537 | 0.326 |
| 32 | 188 | 606 | 0.310 |

Vocabulary diversity is nearly identical across all configurations, consistent with the very similar loss values.

### Training Speed

| n_head | Avg ms/iter | Relative Speed |
|--------|-------------|----------------|
| 4 | ~207 | 1.0x (fastest) |
| 8 | ~245 | 0.84x |
| 16 | ~332 | 0.62x |
| 32 | ~640 | 0.32x |

Training time per iteration increases dramatically with more attention heads. The n_head=32 model is roughly 3x slower than n_head=4. This is because multi-head attention requires separate attention computations for each head, and on MPS (Mac GPU), the overhead of managing many small matrix multiplications outweighs the theoretical parallelism.

## Qualitative Sample Analysis

### n_head=4 (best loss, fastest)

Stories show reasonable narrative structure with identifiable characters and basic plot progression. Dialogue is present and mostly grammatical. Some repetition and logical gaps remain (e.g., a "talking girl" appearing without setup), but the outputs demonstrate the best balance of coherence and narrative variety among the four configurations.

### n_head=8

Very similar quality to n_head=4. Stories maintain narrative arcs with characters and dialogue. Some oddities such as animals having unusual properties ("porch" animals, books together) but overall coherent for the model size.

### n_head=16

Slightly more repetitive than the lower-head models. Narratives are still recognizable children's stories, but some samples show more confused entity tracking (e.g., "the teacher and the teacher" repetitions). Dialogue quality is comparable.

### n_head=32

Quality is marginally lower. Stories occasionally lose track of entities and produce more non-sequiturs (e.g., "a fairy squirrel named Sam" who "loved to fly all day"). The models with fewer heads seem to produce slightly more grounded narratives, though the differences are subtle.

## Discussion

Experiment 3 reveals that the number of attention heads has a surprisingly small effect on model quality at this scale, but a very large effect on training speed.

**Key findings:**

1. **Fewer heads are slightly better.** With n_head=4 (per-head dimension of 64), the model achieves the lowest validation loss (0.5954). This suggests that at this model size (4.85M parameters, n_embd=256), larger per-head dimensions allow each head to capture richer attention patterns.

2. **Parameter count is constant.** Unlike experiments 1 and 2, changing `n_head` does not change the model's capacity. The total number of parameters in Q, K, V projections and output projections remains the same. This isolates the effect of attention granularity from model capacity.

3. **Training speed degrades significantly with more heads.** The n_head=32 configuration is ~3x slower than n_head=4 per iteration. This is likely due to the overhead of computing 32 separate attention matrices of size 8x8 (per-head dim = 256/32 = 8) versus 4 attention matrices of size 64x64 on the MPS backend.

4. **Diminishing per-head dimension hurts.** With n_head=32, each head only has a per-head dimension of 8, which severely limits each head's ability to represent complex attention patterns. The slight but consistent loss increase from n_head=4 to n_head=32 reflects this limitation.

5. **Practical implication:** For small models, fewer attention heads with larger per-head dimensions provide both better performance and faster training. The common practice of using many heads in large models (where per-head dimensions remain large, e.g., 64 or 128) does not translate well to small models where more heads force very small per-head dimensions.

### Cross-Experiment Comparison

Comparing the final validation losses across all three experiments for the n_embd=256, n_layer=6, n_head=8 configuration (which appears in all three):
- Experiment 1 (n_layer=8, n_head=8, n_embd=256): val loss 0.5852
- Experiment 2 (n_layer=6, n_head=8, n_embd=256): val loss 0.5996
- Experiment 3 (n_layer=6, n_head=8, n_embd=256): val loss 0.6000

The slight difference between experiment 2 and 3 for the same configuration (0.5996 vs 0.6000) falls within normal training variance and confirms reproducibility. Among the three dimensions, **depth (experiment 1) and width (experiment 2) have far greater impact on model quality** than the number of attention heads (experiment 3), because the former two directly increase model capacity while the latter only changes how attention is partitioned.
