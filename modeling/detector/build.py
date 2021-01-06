# encoding: utf-8


from ..utils.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for model
"""


def build_model(cfg):
    """
    Build a model from `cfg.MODEL.NAME`.
    Returns:
        an instance of :class:`Model`
    """

    model_name = cfg.MODEL.NAME
    model = MODEL_REGISTRY.get(model_name)(cfg)
    return model
