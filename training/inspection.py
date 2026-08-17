from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from ..configs import baseConfig
from ..data.tokenizer import train_tokenizer
from ..models.base.transformer import BaseTransformer
import torch


config = baseConfig()
tokenizer = train_tokenizer(config.datasets,config.vocab_size)

device = torch.device("cuda")

model = BaseTransformer(
    d_model=config.d_model,
    n_heads=config.heads,
    dropout=config.dropout,
    num_layers=config.layers,
    vocab_size=tokenizer.get_vocab_size(),
    max_seq_len=768,
).to(device)

ROOT = Path(__file__).resolve().parent.parent
path = ROOT / "outputs" / "best_new_blend.pt"

checkpoint = torch.load(path, map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()

# Special token IDs
eos_id = tokenizer.token_to_id("<eos>")
max_length = 768

# Prompt
prompt = "In order to understand climate change, it is important to"
input_ids = torch.tensor(
    tokenizer.encode(prompt).ids,
    dtype=torch.long
).unsqueeze(0).to(device)




def inspect_pipeline(model, input_ids, d_model):
    """Check model health: residual weight scaling + activation stability."""
    L = len(model.decoder_blocks)
    traces = {}
    handles = []

    # Hooks to grab hidden states
    def hook(name):
        def fn(module, inp, out):
            traces[name] = (out[0] if isinstance(out, tuple) else out).detach().cpu().float()
        return fn

    handles.append(model.embedding.register_forward_hook(hook('embed')))
    for i in range(L):
        handles.append(model.decoder_blocks[i].register_forward_hook(hook(f'block_{i+1}')))
    handles.append(model.final_ln.register_forward_hook(hook('final_ln')))
    handles.append(model.output.register_forward_hook(hook('logits')))

    # Run forward (same FP16 as inference)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        model(input_ids, True)
    for h in handles: h.remove()

    # Manually compute the scaled + positional stages
    raw = traces['embed']
    scaled = raw * (d_model ** 0.5)
    
    pipeline = {'embed_raw': raw, 'embed_scaled': scaled}
    for i in range(L): pipeline[f'block_{i+1}'] = traces[f'block_{i+1}']
    pipeline['final_ln'] = traces['final_ln']

    # ---- 1. WEIGHT CHECK (catches missing negative exponent) ----
    exp_std = 0.02 * (2 * L) ** -0.5
    a_std = model.decoder_blocks[0].attn.out_proj.weight.std().item()
    f_std = model.decoder_blocks[0].ffn[-1].weight.std().item()
    print(f"\n[WEIGHTS] attn_proj={a_std:.4f}, ffn_proj={f_std:.4f}  (expected ~{exp_std:.4f})")
    if a_std > exp_std * 2:
        print(">>> WARNING: Residual weights are ~15x too large (missing '** -0.5' in init?)")

    # ---- 2. ACTIVATION TABLE (watch the "Ratio" column) ----
    print("\n{:<12} {:>7} {:>10} {:>8}".format("Stage", "Std", "L2 Norm", "Ratio"))
    print("-" * 42)
    prev_norm = None
    bad_ratio = False
    for name, t in pipeline.items():
        std = t.std().item()
        norm = t.norm().item()
        ratio = norm / prev_norm if prev_norm else 1.0
        flag = " ⚠️ " if (prev_norm and (ratio > 1.3 or ratio < 0.7)) else ""
        if ratio > 1.3: bad_ratio = True
        print(f"{name:<12} {std:>7.3f} {norm:>10.2f} {ratio:>7.2f}{flag}")
        prev_norm = norm

    # ---- 3. FINAL VERDICT ----
    print("-" * 42)
    print(f"Final LayerNorm: mean={traces['final_ln'].mean():.3f}, std={traces['final_ln'].std():.3f}")
    if bad_ratio:
        print(">>> FAIL: L2 Norm Ratio > 1.3 — residual branch is exploding (fix your init).")
    else:
        print(">>> PASS: Pipeline is stable.")

inspect_pipeline(model, input_ids, config.d_model)
