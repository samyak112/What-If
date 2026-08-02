import torch
from torch import nn

from ...utils import make_sinusoidal
from .decoder import DecoderBlock


class BaseTransformer(nn.Module):
    def __init__(self,d_model,n_heads,dropout,num_layers,vocab_size,max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.decoder_blocks = nn.ModuleList(
            DecoderBlock(
                d_model,
                n_heads,
                dropout,
                num_layers=num_layers
            )
            for _ in range(num_layers)
        )
        self.register_buffer(
            "positional_encoding",
            make_sinusoidal(max_seq_len, dim=d_model
                            ),
            persistent=False,
        )

        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(max_seq_len, max_seq_len, dtype=torch.bool),
                diagonal=1,
            ),
            persistent=False,
        )
        self.final_ln = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.embedding.weight

    def forward(self, x,is_attn_mask=False):
        seq_len = x.size(1)

        x = self.embedding(x) * (self.embedding.embedding_dim ** 0.5)   # * sqrt(d_model)
        x = x + self.positional_encoding[:seq_len].unsqueeze(0)

        if(is_attn_mask):
            attn_mask = self.causal_mask[:seq_len, :seq_len]

        else:
            attn_mask = None

        for layer in self.decoder_blocks:
            x = layer(
                x,
                attn_mask,
            )

        x = self.final_ln(x)
        logits = self.output(x)

        return logits



    