import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self,d_model,n_heads,dropout:float,num_layers):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Scale for residual projections
        scale = (2 * num_layers) ** -0.5

        # 1. Initialise ALL linear layers normally
        for module in self.modules():
            if isinstance(module, nn.Linear) and module not in (self.attn.out_proj, self.ffn[-1]):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # 2. Initialise the QKV projection (it’s a Parameter, not a Linear layer)
        nn.init.normal_(self.attn.in_proj_weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.attn.in_proj_bias)

        # 3. Apply the scaled init to the two residual projections
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