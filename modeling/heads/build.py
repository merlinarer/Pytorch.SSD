# encoding: utf-8


from ..utils.registry import Registry

HEAD_REGISTRY = Registry("HEAD")
HEAD_REGISTRY.__doc__ = """
Registry for backbones, which extract feature maps from images
The registered object must be a callable that accepts two arguments:
1. A :class:`detectron2.config.CfgNode`
2. A :class:`detectron2.layers.ShapeSpec`, which contains the input shape specification.
It must returns an instance of :class:`Backbone`.
"""


def build_head(cfg):
    """
    Build a head from `cfg.MODEL.BACKBONE.NAME`.
    Returns:
        an instance of :class:`Backbone`
    """

    head_name = cfg.MODEL.HEADS.NAME
    head = HEAD_REGISTRY.get(head_name)(cfg)
    return head
