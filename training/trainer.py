from pathlib import Path

import torch
from tqdm import tqdm

import wandb

from .checkpoint import save_checkpoint


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loaders,
        optimizer,
        scheduler,
        criterion,
        config,
        checkpoint_path,
        device,
        tokenizer,
        eval_every=2_000,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loaders = val_loaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.config = config
        self.device = device
        self.tokenizer = tokenizer
        self.eval_every = eval_every

        # key -> mixture weight, e.g. "HuggingFaceFW/fineweb/sample-10BT" -> 0.8
        self.dataset_weights = {
            f"{name}/{cfg}": weight for name, cfg, weight in config.datasets
        }
        total_weight = sum(self.dataset_weights.values())
        self.dataset_weights = {k: w / total_weight for k, w in self.dataset_weights.items()}

        # one checkpoint path per tracked criterion: each dataset + the blend
        checkpoint_path = Path(checkpoint_path)
        self.checkpoint_paths = {
            key: checkpoint_path.with_stem(f"{checkpoint_path.stem}_{key.replace('/', '_')}")
            for key in self.dataset_weights
        }
        self.checkpoint_paths["blend"] = checkpoint_path.with_stem(f"{checkpoint_path.stem}_blend")

        self.global_step = 0
        self.best_val_loss = {key: float("inf") for key in self.checkpoint_paths}
        self.prev_val_loss = None

    def validate(self):
        self.model.eval()
        losses = {}

        with torch.no_grad():
            for key, loader in self.val_loaders.items():
                total_loss = 0.0

                for input_ids, output_ids in loader:
                    input_ids = input_ids.to(self.device)
                    output_ids = output_ids.to(self.device)

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = self.model(input_ids, True)
                        loss = self.criterion(
                            logits.view(-1, self.tokenizer.get_vocab_size()),
                            output_ids.view(-1),
                        )

                    total_loss += loss.item()

                losses[key] = total_loss / len(loader)

        self.model.train()

        # blend weighted by each dataset's mixture proportion (config.datasets weight)
        losses["blend"] = sum(
            losses[key] * weight for key, weight in self.dataset_weights.items()
        )

        return losses

    def train_step(self, batch):
        input_ids, output_ids = batch
        input_ids = input_ids.to(self.device)
        output_ids = output_ids.to(self.device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.model(input_ids, True)
            loss = self.criterion(
                outputs.view(-1, self.tokenizer.get_vocab_size()),
                output_ids.view(-1),
            )

        loss.backward()

        if self.global_step % 10 == 0:
            grad_unclipped = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    grad_unclipped += p.grad.data.norm(2).item() ** 2
            grad_unclipped = grad_unclipped ** 0.5
            wandb.log({"debug/grad_norm_unclipped": grad_unclipped}, step=self.global_step)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        return loss.item()

    def train(self):
        self.model.train()
        global_pbar = tqdm(total=self.config.max_steps, desc="Training", unit="step")

        while self.global_step < self.config.max_steps:
            for batch in self.train_loader:
                loss_value = self.train_step(batch)

                self.global_step += 1
                global_pbar.update(1)

                if self.global_step % 100 == 0:
                    wandb.log(
                        {
                            "train/loss": loss_value,
                            "train/lr": self.scheduler.get_last_lr()[0],
                        },
                        step=self.global_step,
                    )

                if self.global_step % self.eval_every == 0:
                    val_losses = self.validate()
                    blend_loss = val_losses["blend"]

                    log_payload = {f"val/loss/{key}": v for key, v in val_losses.items()}

                    if self.prev_val_loss is not None:
                        val_loss_delta = self.prev_val_loss - blend_loss
                        log_payload["val/loss_delta"] = val_loss_delta
                        log_payload["val/loss_delta_per_step"] = val_loss_delta / self.eval_every

                    wandb.log(log_payload, step=self.global_step)

                    self.prev_val_loss = blend_loss

                    # each dataset + the blend is tracked and checkpointed independently
                    for key, loss_value in val_losses.items():
                        if loss_value < self.best_val_loss[key]:
                            self.best_val_loss[key] = loss_value
                            save_checkpoint(
                                self.model,
                                self.optimizer,
                                self.scheduler,
                                self.global_step,
                                self.best_val_loss[key],
                                self.checkpoint_paths[key],
                            )

                if self.global_step >= self.config.max_steps:
                    break

        global_pbar.close()