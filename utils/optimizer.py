import torch.optim as optim


def build_optimizer(cfg, model):
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=cfg.SOLVER.BASE_LR,
                              momentum=cfg.SOLVER.MOMENTUM,
                              weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.SOLVER.BASE_LR,
                               weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        raise RuntimeError
    return optimizer
