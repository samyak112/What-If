from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

import wandb

from ..configs import baseConfig
from ..data.dataloader import build_dataset
from ..models.base.transformer import BaseTransformer
from .scheduler import build_scheduler
from .trainer import Trainer

ROOT = Path(__file__).resolve().parent.parent

device = torch.device("cuda")
run_name = datetime.now().strftime("%b %d, %Y %I:%M:%S %p")
config = baseConfig()

train_loader, val_loaders, total_tokens, tokenizer = build_dataset(config=config)

checkpoint_path = ROOT / "outputs" / "wrong_init.pt"

wandb.init(
    project="what-if",
    name=run_name,
    config={**asdict(config), "total_tokens": total_tokens},
)

model = BaseTransformer(
    d_model=config.d_model,
    n_heads=config.heads,
    dropout=config.dropout,
    num_layers=config.layers,
    vocab_size=tokenizer.get_vocab_size(),
    max_seq_len=config.context_length,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.lr,
    weight_decay=config.weight_decay,
)

scheduler = build_scheduler(optimizer, config)

wandb.watch(model, log="gradients", log_freq=500)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loaders=val_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    config=config,
    checkpoint_path=checkpoint_path,
    device=device,
    tokenizer=tokenizer,
    eval_every=2_000,
)

trainer.train()

wandb.finish()