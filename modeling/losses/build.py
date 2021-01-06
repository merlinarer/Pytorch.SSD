# encoding: utf-8


from ..utils.registry import Registry

LOSS_REGISTRY = Registry("LOSS")
LOSS_REGISTRY.__doc__ = """
Registry for Loss
"""


def build_loss(cfg):
    """
    Build a Loss from `cfg.MODEL.Loss.NAME`.
    Returns:
        an instance of :class:`Loss`
    """

    loss_name = cfg.MODEL.LOSSES.NAME
    loss = LOSS_REGISTRY.get(loss_name)(cfg)
    return loss
