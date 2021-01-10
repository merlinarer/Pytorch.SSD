import torch

def smooth_l1(pred, target, beta=1.0,weight=None, reduction='mean', avg_factor=None):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    if weight!=None:
        loss *= weight
    if reduction=='mean':
        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        else:
            loss = loss.mean()
    return loss
