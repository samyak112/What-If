from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)


def build_scheduler(optimizer, config):
    warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=config.warmup_steps,      
    )

    hold = ConstantLR(
        optimizer,
        factor=1.0,
        total_iters=config.hold_steps,        
    )

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=config.max_steps - config.warmup_steps - config.hold_steps,
        eta_min=1e-5,                        
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, hold, cosine],
        milestones=[config.warmup_steps, config.warmup_steps + config.hold_steps],
    )

    return scheduler