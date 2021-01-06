# encoding: utf-8


from ..utils.registry import Registry

HEAD_REGISTRY = Registry("HEAD")
HEAD_REGISTRY.__doc__ = """
Registry for head
"""


def build_head(cfg):
    """
    Build a head from `cfg.MODEL.HEAD.NAME`.
    Returns:
        an instance of :class:`Head`
    """

    head_name = cfg.MODEL.HEADS.NAME
    head = HEAD_REGISTRY.get(head_name)(cfg)
    return head
