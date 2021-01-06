from .detector import build_model


def build_detector(cfg):
    return build_model(cfg)


if __name__ == '__main__':
    import sys
    sys.path.append('/home/workspace/merlin/projects/SSD')
    from config import cfg
    model = build_detector(cfg)
