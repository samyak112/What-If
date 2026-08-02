from pathlib import Path

import torch
from transformers import AutoTokenizer

from ..configs import baseConfig
from ..models.base.transformer import BaseTransformer

config = baseConfig()

tokenizer = AutoTokenizer.from_pretrained("gpt2",use_fast=True)


device = torch.device("cuda")

model = BaseTransformer(
    d_model=config.d_model,
    n_heads=config.heads,
    dropout=config.dropout,
    num_layers=config.layers,
    vocab_size=tokenizer.vocab_size,
    max_seq_len=1024,
).to(device)

ROOT = Path(__file__).resolve().parent.parent
path = ROOT / "outputs"  / "lr_fix_checkpoint.pt"

checkpoint = torch.load(path, map_location=device)

model.load_state_dict(checkpoint["model"])
model.eval()

input_ids = torch.tensor(
    tokenizer.encode("If Superman had explored these issues instead of bashing unions"),
    dtype=torch.long
).unsqueeze(0).to(device)

with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
    for _ in range(100):

        logits = model(input_ids,True)

        # Last token logits
        next_token_logits = logits[:, -1, :]

        temperature = 0.8
        k = 10

        logits = next_token_logits / temperature

        values, indices = torch.topk(logits, k)
        probs = torch.softmax(values, dim=-1)

        sample = torch.multinomial(probs, 1)
        next_token = indices.gather(-1, sample)

        # Append token
        input_ids = torch.cat([input_ids, next_token], dim=1)

generated = input_ids[0].tolist()
print(tokenizer.decode(generated))