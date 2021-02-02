import torch


def decode(loc, priors, variances=None):
    """Decode locations from predictions to bb.
    Args:
        loc (tensor): Shape: [num_priors,4]
        priors (tensor): Shape: [num_priors,4].  xywh
        variances: (list[float]) Variances of priorboxes xyxy
    """
    if variances is None:
        variances = [1, 1]
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def encode(matched,
           priors,
           variances=None):
    """Encode locations from bb cords to model output.
        Args:
            matched (tensor): Shape: [num_priors,4]
            priors (tensor): Shape: [num_priors,4].  xywh
            variances: (list[float])
        """
    if variances is None:
        variances = [1, 1]
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    g_cxcy /= (variances[0] * priors[:, 2:])
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]

def encode_xyxy(matched,
           priors,
           variances=None):
    """Encode locations from bb cords to model output.
        Args:
            matched (tensor): Shape: [num_priors,4]  gt xyxy
            priors (tensor): Shape: [num_priors,4].  anchor xyxy
            variances: (list[float])
        """
    if variances is None:
        variances = [1, 1]
    p_xy = (priors[:, :2] +priors[:,2:]) /2
    p_wh = (priors[:, 2:] - priors[:, :2])
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - p_xy
    g_cxcy /= (variances[0] * p_wh)
    g_wh = (matched[:, 2:] - matched[:, :2]) / p_wh
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]