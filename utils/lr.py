import torch


def adjust_learning_rate(cfg, optimizer, step):
    gamma = cfg.SOLVER.GAMMA
    if cfg.SOLVER.WARMUP and step < cfg.SOLVER.WARMUP_ITERS:
        lr = cfg.SOLVER.BASE_LR * step * 1.0 / cfg.SOLVER.WARMUP_ITERS
    elif step in cfg.SOLVER.STEPS:
        idx = list(cfg.SOLVER.STEPS).index(step) + 1
        lr = cfg.SOLVER.BASE_LR * (gamma ** idx)
    else:
        return
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return

def adjust_learning_rate_epoch(cfg, optimizer, step, epoch, it):
    gamma = cfg.SOLVER.GAMMA
    if cfg.SOLVER.WARMUP and step <= cfg.SOLVER.WARMUP_ITERS:
        lr = (1.0 - cfg.SOLVER.LR_RATIO) * cfg.SOLVER.BASE_LR * step * 1.0 / cfg.SOLVER.WARMUP_ITERS + cfg.SOLVER.LR_RATIO * cfg.SOLVER.BASE_LR
    elif epoch in cfg.SOLVER.STEPS_EPOCH and it == 1:

        idx = list(cfg.SOLVER.STEPS_EPOCH).index(epoch) + 1
        lr = cfg.SOLVER.BASE_LR * (gamma ** idx)
    else:
        return
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
