from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from tqdm import tqdm

import wandb

from ..configs import baseConfig
from ..data.dataloader import build_dataset
from ..models.base.transformer import BaseTransformer

ROOT = Path(__file__).resolve().parent.parent

device = torch.device("cuda")
run_name = datetime.now().strftime("%b %d, %Y %I:%M:%S %p")
config = baseConfig()

train_loader, val_loader,total_tokens,tokenizer = build_dataset(config=config)

ROOT = Path(__file__).resolve().parent.parent
checkpoint_path = ROOT / "outputs" / "tokenizer_change_plus_increased_dims_checkpoint.pt"


wandb.init(
    project="what-if",          # pick a project name
    name=run_name,
    config={**asdict(config),"total_tokens":total_tokens},    # this replaces your writer.add_text("config", ...) call
)

model = BaseTransformer(
    d_model=config.d_model,
    n_heads=config.heads,
    dropout=config.dropout,
    num_layers=config.layers,
    vocab_size=tokenizer.get_vocab_size(),
    max_seq_len=1024
    ).to(device)

criterion = nn.CrossEntropyLoss()
optimizer  = torch.optim.AdamW(
    model.parameters(),
    lr=config.lr,
    weight_decay=config.weight_decay
)

total_steps = config.max_steps

warmup = LinearLR(
    optimizer,
    start_factor=0.01,
    end_factor=1.0,
    total_iters=config.warmup_steps,      # e.g., 2000
)

hold = ConstantLR(
    optimizer,
    factor=1.0,
    total_iters=config.hold_steps,        # e.g., 20_000
)

cosine = CosineAnnealingLR(
    optimizer,
    T_max=total_steps - config.warmup_steps - config.hold_steps,
    eta_min=1e-5,                         # slightly lower floor
)

scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup, hold, cosine],
    milestones=[config.warmup_steps, config.warmup_steps + config.hold_steps],
)

best_val_loss = float("inf")

wandb.watch(model, log="gradients", log_freq=500)


def validate():
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for input_ids, output_ids in val_loader:
            input_ids = input_ids.to(device)
            output_ids = output_ids.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids, True)
                loss = criterion(
                    logits.view(-1, tokenizer.get_vocab_size()),
                    output_ids.view(-1),
                )

            total_loss += loss.item()

    model.train()
    return total_loss / len(val_loader)


def save_checkpoint(global_step):
    
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": global_step,
            "best_val_loss": best_val_loss,
        },
        checkpoint_path,
    )

global_step = 0
eval_every = 2_000
prev_val_loss = None

model.train()

global_pbar = tqdm(total=config.max_steps, desc="Training", unit="step")


while global_step < config.max_steps:
    for batch in train_loader:

        input_ids, output_ids = batch
        input_ids = input_ids.to(device)
        output_ids = output_ids.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids, True)
            loss = criterion(
                outputs.view(-1, tokenizer.get_vocab_size()),
                output_ids.view(-1),
            )

        loss.backward()

        # just for logging the trend of gradients
        if global_step % 10 == 0:
            grad_unclipped = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_unclipped += p.grad.data.norm(2).item() ** 2
            grad_unclipped = grad_unclipped ** 0.5
            wandb.log({"debug/grad_norm_unclipped": grad_unclipped}, step=global_step)


        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        global_step += 1
        global_pbar.update(1)

        # log
        if global_step % 100 == 0:
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                },
                step=global_step,
            )

        # validate
        if global_step % eval_every == 0:
            val_loss = validate()

            # Compute decrease since last evaluation (if available)
            if prev_val_loss is not None:
                val_loss_delta = prev_val_loss - val_loss   # positive when loss drops
                val_loss_delta_per_step = val_loss_delta / eval_every

                wandb.log({
                    "val/loss": val_loss,
                    "val/loss_delta": val_loss_delta,                 # absolute drop
                    "val/loss_delta_per_step": val_loss_delta_per_step  # drop per training step
                }, step=global_step)
            else:
                wandb.log({"val/loss": val_loss}, step=global_step)

            # Remember for next time
            prev_val_loss = val_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(global_step)

        if global_step >= config.max_steps:
            break

wandb.finish()