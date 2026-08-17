import torch


def save_checkpoint(model, optimizer, scheduler, global_step, best_val_loss, checkpoint_path):
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


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    ckpt = torch.load(checkpoint_path)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])

    return ckpt["step"], ckpt["best_val_loss"]