from pathlib import Path
import torch
import torch.nn.functional as F

from ..configs import baseConfig
from ..models.base.transformer import BaseTransformer
from ..data.tokenizer import train_tokenizer

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

with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
    for _ in range(200):  # maximum tokens to generate
        logits = model(input_ids, True)
        next_logits = logits[:, -1, :]   # shape: (1, vocab_size)

        # ---- Repetition penalty ----
        # Reduce the logits of tokens already present in the sequence.
        for token_id in set(input_ids[0].tolist()):
            logit = next_logits[0, token_id]
            next_logits[0, token_id] = logit / 1.2 if logit > 0 else logit * 1.2

        # ---- Temperature scaling ----
        temperature = 0.7
        next_logits = next_logits / temperature

        # ---- Top‑p (nucleus) sampling ----
        sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        p = 0.95
        sorted_mask = cumulative_probs > p
        # Shift the mask to keep the first token that exceeds the threshold
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False   # always keep at least one token

        # Scatter the mask back to the original vocabulary order
        mask = torch.zeros_like(next_logits, dtype=torch.bool).scatter_(
            dim=-1, index=sorted_indices, src=sorted_mask
        )
        next_logits[mask] = float('-inf')

        # ---- Sampling ----
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)

        # Stop if EOS is generated
        if next_token.item() == eos_id:
            break

        # Stop if model’s max sequence length is reached
        if input_ids.size(1) >= max_length:
            break

        # Append the new token
        input_ids = torch.cat([input_ids, next_token], dim=1)

# Decode and print
generated_text = tokenizer.decode(input_ids[0].tolist())
print(generated_text)