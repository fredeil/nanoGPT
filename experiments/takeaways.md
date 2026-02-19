# Key Takeaways — Transformer Scaling Experiments

## The One-Liner

At small scale, **how many parameters you have matters far more than how you organize them**.

## What Each Dimension Does

### Depth (n_layer) — adds compositional ability

- Each layer is a new round of processing. Layer 1 might learn character combos, layer 2 learns words, layer 3 learns phrases, etc.
- Parameters scale **linearly** with depth (~1.57M per layer in our setup).
- Going from 2 → 4 layers was a huge jump in quality. Going from 8 → 12 barely mattered.
- Why: TinyStories has simple grammar and short sentences. After ~6-8 layers, the model has enough depth to represent that structure. Extra layers have nothing useful left to learn.
- The shallow model (2 layers) could produce valid words but couldn't string them into coherent sentences. It lacks the compositional depth to plan beyond the next few characters.

### Width (n_embd) — adds representational richness

- Each token gets compressed into an n_embd-dimensional vector. Bigger vector = more information carried per token.
- Parameters scale **quadratically** with width (because MLP weights are n_embd × 4·n_embd). This makes width the most parameter-dense knob.
- Had the **biggest effect** of any experiment — 0.140 loss spread vs 0.122 for depth.
- The narrowest model (n_embd=128) couldn't even form valid words consistently — it produced gibberish like "smooter". The character-level tokenization means the model must use its internal representation to build up words from characters, and 128 dimensions isn't enough.
- Practical insight: if you have a parameter budget, spending it on width gives you more bang for buck than depth.

### Attention Heads (n_head) — changes structure, not capacity

- Splitting attention into more heads does NOT change the parameter count. The Q/K/V matrices stay the same size.
- This makes it a pure test of **how attention is organized**, not how much capacity the model has.
- Result: **almost no effect on quality** (0.011 loss spread). But huge effect on training speed (n_head=32 was 3x slower than n_head=4).
- Why fewer heads won slightly: with n_embd=256 and n_head=32, each head only has 8 dimensions to work with. That's very restrictive. With n_head=4, each head gets 64 dimensions and can represent richer attention patterns.
- The "many heads = more diverse attention" idea from big models (GPT-3 has 96 heads with dim=128 each) doesn't apply when heads become tiny.

## Diminishing Returns

This is a core concept the assignment wants you to understand:

- 2→4 layers: huge gain (0.073 loss drop)
- 4→8 layers: good gain (0.034)
- 8→12 layers: small gain (0.015) despite adding MORE parameters

Same pattern with width. The first units of capacity matter most. After a point, the model has enough capacity for the task and extra parameters just sit there.

This connects to real-world LLM scaling: there's always a tradeoff between model size, training cost, and actual performance gain. Scaling laws (like Chinchilla) try to find the sweet spot.

## Overfitting (or lack thereof)

- We saw NO overfitting in any model. Train and val loss stayed aligned.
- Why: TinyStories is large relative to our biggest model (10.81M params). With dropout=0.0, overfitting would need either longer training or a smaller dataset.
- But there's a subtler form of "overfitting to the distribution": the best models produce very formulaic, template-like stories. They've learned what TinyStories looks like and stick to it. They're not creative — they're good mimics.

## The Diversity Paradox

- The worst models had the HIGHEST vocabulary diversity scores.
- This seems wrong until you read the samples: bad models jump randomly between topics, introducing new words each time. Good models stay on topic and reuse the same words within a story.
- Lesson: **metrics can be misleading without qualitative analysis**. A diversity score alone doesn't tell you if the model is good.

## Concepts to Know for Exam

1. **Transformer block structure**: self-attention + MLP + layer norms. Each layer adds one block. The MLP is the parameter-heavy part (n_embd × 4·n_embd × 2).

2. **Multi-head attention**: the embedding is split into n_head chunks, each doing independent attention. Per-head dimension = n_embd / n_head. More heads = more parallel attention patterns but each pattern is lower-dimensional.

3. **Capacity vs structure**: depth and width add capacity (more parameters). Head count changes structure (same parameters, different organization). Capacity dominates at small scale.

4. **Scaling behavior**: more parameters = better, but with diminishing returns. The improvement per parameter decreases as models get larger.

5. **Character-level vs token-level**: we used character-level tokenization (vocab=228). The model has to learn to spell words before it can learn grammar. This makes the task harder and means narrow models struggle with basic word formation.

6. **Overfitting in language models**: happens when the model memorizes training data instead of learning patterns. We didn't see it because dataset was large relative to model size. Dropout is one way to combat it.

7. **Loss as a metric**: lower validation loss = better generalization. But loss alone doesn't capture everything — qualitative analysis of generated text reveals things like coherence, repetition, and creativity that loss numbers miss.
