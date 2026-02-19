# Experiment 2: Width Sweep (n_embd)

## Setup

This experiment investigates the effect of model width (embedding dimension) on language generation quality. We train four GPT models on the TinyStories dataset, varying only the embedding dimension while keeping all other hyperparameters fixed.

| Hyperparameter | Value |
|---|---|
| `n_layer` | 6 |
| `n_head` | 8 |
| `dropout` | 0.0 |
| `block_size` | 256 |
| `batch_size` | 64 |
| `max_iters` | 5000 |
| `learning_rate` | 1e-3 |
| `min_lr` | 1e-4 |
| `warmup_iters` | 100 |

**Varied:** `n_embd` in {128, 192, 256, 384}

Note: `n_embd` must be divisible by `n_head`. With `n_head=8`, all four values (128, 192, 256, 384) satisfy this constraint, yielding per-head dimensions of 16, 24, 32, and 48 respectively.

## Results

### Parameter Counts

| n_embd | Total Parameters | Decayed | Non-decayed |
|--------|-----------------|---------|-------------|
| 128 | 1.24M | 1,241,600 | 1,664 |
| 192 | 2.75M | 2,747,136 | 2,496 |
| 256 | 4.85M | 4,842,496 | 3,328 |
| 384 | 10.81M | 10,802,688 | 4,992 |

Parameters scale roughly quadratically with width. Doubling `n_embd` from 128 to 256 yields a 3.9x increase in parameters, while going from 128 to 384 (3x width) yields an 8.7x increase. This is because the MLP layers contain weight matrices of size `n_embd x 4*n_embd`, making parameter count scale as O(n_embd^2).

### Validation Loss

| n_embd | Val Loss (step 1000) | Val Loss (step 2500) | Val Loss (final, step 5000) |
|--------|---------------------|---------------------|----------------------------|
| 128 | 1.0461 | 0.7924 | 0.6968 |
| 192 | 0.9237 | 0.7151 | 0.6324 |
| 256 | 0.8534 | 0.6836 | 0.5996 |
| 384 | 0.7747 | 0.6343 | 0.5567 |

### Training Loss Curves

Validation loss at every 250 steps:

| Step | n_embd=128 | n_embd=192 | n_embd=256 | n_embd=384 |
|------|-----------|-----------|-----------|-----------|
| 0 | 5.4493 | 5.4568 | 5.4048 | 5.4898 |
| 250 | 2.0487 | 1.8467 | 1.7011 | 1.4109 |
| 500 | 1.4274 | 1.2234 | 1.1135 | 0.9942 |
| 750 | 1.1702 | 1.0230 | 0.9492 | 0.8447 |
| 1000 | 1.0461 | 0.9237 | 0.8534 | 0.7747 |
| 1250 | 0.9633 | 0.8657 | 0.7977 | 0.7348 |
| 1500 | 0.9116 | 0.8136 | 0.7594 | 0.7078 |
| 1750 | 0.8750 | 0.7791 | 0.7346 | 0.6854 |
| 2000 | 0.8450 | 0.7496 | 0.7134 | 0.6660 |
| 2250 | 0.8159 | 0.7298 | 0.6930 | 0.6476 |
| 2500 | 0.7924 | 0.7151 | 0.6836 | 0.6343 |
| 2750 | 0.7759 | 0.7039 | 0.6630 | 0.6216 |
| 3000 | 0.7571 | 0.6883 | 0.6532 | 0.6111 |
| 3250 | 0.7478 | 0.6776 | 0.6452 | 0.5988 |
| 3500 | 0.7307 | 0.6680 | 0.6353 | 0.5903 |
| 3750 | 0.7252 | 0.6585 | 0.6232 | 0.5819 |
| 4000 | 0.7163 | 0.6494 | 0.6170 | 0.5729 |
| 4250 | 0.7101 | 0.6425 | 0.6094 | 0.5652 |
| 4500 | 0.7025 | 0.6373 | 0.6057 | 0.5640 |
| 4750 | 0.6989 | 0.6350 | 0.6033 | 0.5607 |
| 5000 | 0.6968 | 0.6324 | 0.5996 | 0.5567 |

All models show consistent improvement with no overfitting throughout training.

### Vocabulary Diversity

Measured as unique_words / total_words across 5 generated samples (1000 tokens each):

| n_embd | Total Words | Unique Words | Diversity Ratio |
|--------|-------------|--------------|-----------------|
| 128 | 1011 | 272 | 0.2690 |
| 192 | 1010 | 294 | 0.2911 |
| 256 | 1037 | 319 | 0.3076 |
| 384 | 1026 | 308 | 0.3002 |

Vocabulary diversity increases with width up to `n_embd=256`, then slightly decreases at 384. The wider models may converge on more formulaic but coherent story structures, slightly reducing lexical variety.

### Training Speed

| n_embd | ms/iter | Approx. Total Time |
|--------|---------|-------------------|
| 128 | ~153 | ~13 min |
| 192 | ~209 | ~18 min |
| 256 | ~245 | ~21 min |
| 384 | ~435 | ~38 min |

Training time scales super-linearly with width due to the quadratic growth of parameter count.

### Qualitative Sample Analysis

**n_embd=128 (1.24M params):** Text is largely incoherent. Sentences are grammatically fragile with frequent errors: "They kind, they see a cabinet on the food." Narratives lack any sustained thread — characters, objects, and actions appear randomly. Made-up or garbled words appear occasionally: "cabins", "smooter", "builly."

**n_embd=192 (2.75M params):** Moderate improvement. Sentences are more grammatically correct and stories maintain a basic theme for several sentences. However, logic remains weak: "The sun was happy to have and enjoyed the sun with TV and Sam." Characters sometimes behave inconsistently, and pronoun references are often confused.

**n_embd=256 (4.85M params):** A clear qualitative jump. Stories have recognizable narrative arcs with beginnings, middles, and endings. Dialogue is more natural and contextually appropriate. Moral statements appear at story ends, mimicking the TinyStories format: "The moral of the story is: it is better than being so kind..." Some awkward phrasing persists but overall coherence is substantially improved.

**n_embd=384 (10.81M params):** Best overall quality. Stories are coherent with well-maintained character identities and consistent settings. Dialogue flows naturally between characters. The model produces more complex narrative structures including unexpected events: "But then, something unexpected happened." However, some repetition still occurs, and occasional logical errors persist.

## Discussion

- **Width strongly impacts loss:** Each increase in `n_embd` produces clear improvement, from 0.6968 (128) to 0.5567 (384). The total spread (0.1401) is larger than the depth sweep's spread (0.1217 in Experiment 1), suggesting width may be a more impactful lever than depth for this model size range.
- **Quadratic parameter cost:** Unlike depth which scales linearly, width scales quadratically in parameters. Going from `n_embd=128` to 384 increases parameters by 8.7x, compared to depth's 5.6x for a similar loss improvement. Width is more parameter-expensive per unit of loss reduction.
- **Early convergence advantage of wider models:** At step 250, the `n_embd=384` model (val loss 1.41) is already far ahead of `n_embd=128` at the same step (val loss 2.05). Wider models learn faster in the early phase, likely because the richer representations can capture more patterns per gradient update.
- **No overfitting:** As with the depth sweep, all models maintain close train/val loss alignment, confirming sufficient dataset size relative to model capacity.
- **Qualitative gains are dramatic:** The jump from 128 to 256 embedding dimension represents the most significant qualitative improvement, taking the model from largely incoherent text to recognizable story structures. The further improvement to 384 is more incremental in quality.
- **Efficiency comparison with depth:** The `n_layer=8` model from Experiment 1 (6.42M params, val loss 0.5852) achieves comparable loss to `n_embd=256` here (4.85M params, val loss 0.5996) but with 32% more parameters. Meanwhile, `n_embd=384` (10.81M params, val loss 0.5567) outperforms `n_layer=12` (9.57M params, val loss 0.5703). This suggests that at these scales, width may be slightly more efficient than depth per parameter, though the relationship is complex.
