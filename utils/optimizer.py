import torch.optim as optim


def build_optimizer(optimizer_name,
                    model,
                    base_lr,
                    momentum,
                    wd):
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=base_lr,
                              momentum=momentum,
                              weight_decay=wd)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=base_lr,
                               weight_decay=wd)

    else:
        raise RuntimeError

    return optimizer
