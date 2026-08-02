import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self,d_model,n_heads,dropout:float,num_layers):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # ---- 1. First, initialise ALL linear layers to a clean baseline ----
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # ---- 2. Then apply the special scaled init to the residual projections ----
        scale = (2 * num_layers) ** -0.5
        nn.init.normal_(self.attn.out_proj.weight, mean=0.0, std=0.02 * scale)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.normal_(self.ffn[-1].weight, mean=0.0, std=0.02 * scale)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, x, attn_mask=None):
        attn_input = self.layer_norm_1(x)

        attn_output, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            attn_mask=attn_mask,
            need_weights=False,
        )

        # residual connection
        x = x + self.dropout(attn_output)
        ffn_input = self.layer_norm_2(x)
        ffn_out = self.ffn(ffn_input)

        x = x + self.dropout(ffn_out)

        return x