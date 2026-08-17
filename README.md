# What-If

Experiments on a small decoder-only Transformer — how small can it be and still train well, what actually reduces loss, what doesn't.

Inspired by Microsoft's [TinyStories](https://arxiv.org/abs/2305.07759), which showed sub-10M-param models can write coherent text if the *data* is simplified.

## Architecture

Pre-LN decoder-only Transformer. `d_model=384, heads=8, layers=10, context_length=768, vocab_size=16384` (~24-30M params). Sinusoidal PE, GELU FFN (4x), weight-tied embeddings, GPT-2-style scaled residual init.

## Data

BPE tokenizer trained from scratch. 80% FineWeb / 20% WikiText-103, streamed and tokenized to `.bin` files, token budget = `20 * param_count`. Val loss tracked per-dataset and as a blend.

## Training

AdamW, bf16, warmup → hold → cosine LR, grad clipping, logged to `wandb`.

## Diagnostics used

- **Overfit test**: one batch, 200 steps, should hit ~zero loss (checks for structural bugs)
- **update_ratio** = `(lr × grad_norm) / param_norm` — healthy range 1e-4–5e-3; below that means the optimizer isn't really moving weights
- Watch `grad_norm_unclipped`, `param_norm_total`, and loss curve shape together, not alone

## Findings

**Fixed the model:**
- Proper init (all linears `N(0,0.02²)`, residual projections scaled by `(2·layers)^-0.5`) — val loss 4.96 → 4.29 @ 52k
- Fixed embedding/PE scale mismatch (embeddings were ~30x weaker than sinusoidal PE, model leaned on position only) — 5.6 → 5.1 @ 4k
- Relaxed grad clip 1.0 → 5.0 — clipping was silently overriding the LR schedule (gradients spiking to 4-50 were all crushed to the same step size)
- Warmup → hold(20k) → cosine, peak LR 3e-4 → 6e-4 — 7.33 → 6.74 @ 500 steps total

**Didn't help (or unclear):**
- Dropout 0.1 + wd 0.1 together: hurt, but confounded — two vars changed at once, needs re-testing separately
- Removing LR hold: no measurable difference (possibly real, possibly not tested long enough)
- Selective weight decay: no difference, but only run to 5k steps — inconclusive
- SwiGLU: no improvement — unclear if FFN params were matched to the GELU baseline first
