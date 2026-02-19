# Experiment 1: Depth Sweep (n_layer)

## Setup

This experiment investigates the effect of model depth on language generation quality. We train four GPT models on the TinyStories dataset, varying only the number of transformer layers while keeping all other hyperparameters fixed.

| Hyperparameter | Value |
|---|---|
| `n_embd` | 256 |
| `n_head` | 8 |
| `dropout` | 0.0 |
| `block_size` | 256 |
| `batch_size` | 64 |
| `max_iters` | 5000 |
| `learning_rate` | 1e-3 |
| `min_lr` | 1e-4 |
| `warmup_iters` | 100 |

**Varied:** `n_layer` in {2, 4, 8, 12}

## Results

### Parameter Counts

| n_layer | Total Parameters | Decayed | Non-decayed |
|---------|-----------------|---------|-------------|
| 2 | 1.70M | 1,696,768 | 1,280 |
| 4 | 3.27M | 3,269,632 | 2,304 |
| 8 | 6.42M | 6,415,360 | 4,352 |
| 12 | 9.57M | 9,561,088 | 6,400 |

Parameters scale linearly with depth, adding approximately 1.57M parameters per layer (each layer contributes self-attention weights, MLP weights, and layer norm parameters).

### Validation Loss

| n_layer | Val Loss (step 1000) | Val Loss (step 2500) | Val Loss (final, step 5000) |
|---------|---------------------|---------------------|----------------------------|
| 2 | 0.9693 | 0.7783 | 0.6920 |
| 4 | 0.8857 | 0.7010 | 0.6193 |
| 8 | 0.8407 | 0.6638 | 0.5852 |
| 12 | 0.8348 | 0.6509 | 0.5703 |

### Training Loss Curves

Validation loss at every 250 steps:

| Step | n_layer=2 | n_layer=4 | n_layer=8 | n_layer=12 |
|------|-----------|-----------|-----------|------------|
| 0 | 5.5409 | 5.4914 | 5.4746 | 5.4076 |
| 250 | 1.8388 | 1.6812 | 1.6527 | 1.6811 |
| 500 | 1.2310 | 1.1343 | 1.0945 | 1.1089 |
| 750 | 1.0546 | 0.9763 | 0.9280 | 0.9218 |
| 1000 | 0.9693 | 0.8857 | 0.8407 | 0.8348 |
| 1250 | 0.8958 | 0.8255 | 0.7827 | 0.7752 |
| 1500 | 0.8551 | 0.7867 | 0.7449 | 0.7355 |
| 1750 | 0.8311 | 0.7559 | 0.7159 | 0.7162 |
| 2000 | 0.8129 | 0.7339 | 0.7026 | 0.6856 |
| 2250 | 0.7975 | 0.7193 | 0.6768 | 0.6664 |
| 2500 | 0.7783 | 0.7010 | 0.6638 | 0.6509 |
| 2750 | 0.7657 | 0.6893 | 0.6507 | 0.6364 |
| 3000 | 0.7529 | 0.6762 | 0.6332 | 0.6248 |
| 3250 | 0.7378 | 0.6669 | 0.6289 | 0.6159 |
| 3500 | 0.7281 | 0.6554 | 0.6202 | 0.6044 |
| 3750 | 0.7182 | 0.6460 | 0.6075 | 0.5972 |
| 4000 | 0.7100 | 0.6364 | 0.6038 | 0.5864 |
| 4250 | 0.7037 | 0.6299 | 0.5954 | 0.5822 |
| 4500 | 0.6990 | 0.6282 | 0.5906 | 0.5764 |
| 4750 | 0.6926 | 0.6230 | 0.5890 | 0.5711 |
| 5000 | 0.6920 | 0.6193 | 0.5852 | 0.5703 |

All models show consistent improvement throughout training with no signs of overfitting (train and val losses remain closely aligned), likely due to `dropout=0.0` combined with the large TinyStories dataset relative to model size.

### Vocabulary Diversity

Measured as unique_words / total_words across 5 generated samples (1000 tokens each):

| n_layer | Total Words | Unique Words | Diversity Ratio |
|---------|-------------|--------------|-----------------|
| 2 | 1031 | 322 | 0.3123 |
| 4 | 1006 | 266 | 0.2644 |
| 8 | 1035 | 279 | 0.2696 |
| 12 | 1061 | 275 | 0.2592 |

The 2-layer model actually shows the highest diversity ratio. This is somewhat counterintuitive but can be explained: the shallower model produces more incoherent text, jumping between topics and using unrelated words, which inflates the diversity metric. Deeper models generate more focused, on-topic stories which naturally reuse story-relevant vocabulary (character names, common verbs, connectors).

### Training Speed

| n_layer | ms/iter | Approx. Total Time |
|---------|---------|-------------------|
| 2 | ~88 | ~8 min |
| 4 | ~175 | ~15 min |
| 8 | ~345 | ~29 min |
| 12 | ~530 | ~44 min |

Training time scales roughly linearly with depth, as expected.

### Qualitative Sample Analysis

**n_layer=2 (1.70M params):** Generates grammatically weak text with frequent topic drift. Stories start coherently but quickly lose narrative thread. Characters and objects appear and disappear without logic. Dialogue is sometimes attributed incorrectly. Example issues: "The tower greated a nose", "She took a steps and put them in the puddle too."

**n_layer=4 (3.27M params):** Noticeable improvement in sentence structure and basic narrative flow. Stories maintain a theme for longer stretches. Dialogue is more natural: "Can you help us?" / "No, dog. You are not safe." However, stories still contain logical inconsistencies and occasionally lose coherence.

**n_layer=8 (6.42M params):** Further improvement in narrative coherence. Stories maintain consistent characters throughout. Dialogue feels more purposeful. The model produces more natural story structures with beginnings, conflicts, and resolutions, though some awkward phrasing remains: "I replace", "He felt tickling his toys."

**n_layer=12 (9.57M params):** Best overall coherence and narrative structure. Stories have clear beginnings, middles, and endings. Dialogue is contextually appropriate. The model demonstrates understanding of story conventions (character introductions, settings, morals). However, it still occasionally drifts: cookies becoming a topic of extended discussion, or character roles shifting mid-story.

## Discussion

- **Depth clearly improves loss:** Each doubling of layers produces measurable improvement in validation loss, from 0.6920 (2 layers) to 0.5703 (12 layers).
- **Diminishing returns:** The biggest loss improvement comes from 2 to 4 layers (delta of 0.0727), while 8 to 12 layers yields only 0.0149. This suggests diminishing returns from added depth at this model width.
- **No overfitting observed:** All models maintain close train/val loss alignment through 5000 iterations, indicating the dataset is sufficiently large for these model sizes.
- **Qualitative gains are substantial:** While the quantitative loss differences between 8 and 12 layers are small, the qualitative difference in text coherence is meaningful, particularly in narrative structure and dialogue consistency.
- **Cost-benefit tradeoff:** The 4-layer model offers arguably the best cost-benefit ratio, achieving strong loss improvement (0.6193) at only 3.27M parameters and ~15 min training time, compared to the 12-layer model's marginal improvement at 5.6x the parameters and ~3x the training time.
