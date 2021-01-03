import torch


def adjust_learning_rate(cfg, optimizer, gamma, step):
    if cfg.SOLVER.WARMUP and step < cfg.SOLVER.WARMUP_ITERS:
        lr = cfg.SOLVER.BASE_LR * step * 1.0 / cfg.SOLVER.WARMUP_ITERS
    elif step in cfg.SOLVER.STEPS:
        idx = list(cfg.SOLVER.STEPS).index(step) + 1
        lr = cfg.SOLVER.BASE_LR * (gamma ** idx)
    else:
        return
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
