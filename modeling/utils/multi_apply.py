import torch
from functools import partial
from six.moves import map, zip

def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results))) #转置矩阵后划分

def image_to_level(target, num_anchors_level):
    """
    Turn [anchor_img0, anchor_img1..] to [anchor_level0, anchor_level2..]
         List length=batch size           List length=level length
         (num_anchors,4)                  (batch_size, num_level_anchors, 4)
    :param target:
    :param num_anchors_lever:
    :return:
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_anchors_level:
        end = start + n
        # level_targets.append(target[:, start:end].squeeze(0))
        level_targets.append(target[:, start:end])
        start = end
    return level_targets