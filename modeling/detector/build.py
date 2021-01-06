# encoding: utf-8


from ..utils.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for backbones, which extract feature maps from images
The registered object must be a callable that accepts two arguments:
1. A :class:`detectron2.config.CfgNode`
2. A :class:`detectron2.layers.ShapeSpec`, which contains the input shape specification.
It must returns an instance of :class:`Backbone`.
"""


def build_model(cfg):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.
    Returns:
        an instance of :class:`Backbone`
    """

    model_name = cfg.MODEL.NAME
    model = MODEL_REGISTRY.get(model_name)(cfg)
    return model
