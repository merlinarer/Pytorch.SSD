from .cross_entroy_loss import cross_entropy_loss, log_accuracy
from .focal_loss import focal_loss
from .triplet_loss import triplet_loss
from .circle_loss import circle_loss


def get_loss(cfg, outs, label):
    r"""
    Compute loss from modeling's outputs, the loss function input arguments
    must be the same as the outputs of the model forwarding.
    """
    # fmt: off
    # outputs         = outs["outputs"]
    outputs = outs
    gt_labels = label
    # cls_outputs           = outs
    # print(outs.shape)
    # gt_labels         = label
    # model predictions
    pred_class_logits = outputs['pred_class_logits'].detach()
    cls_outputs = outputs['cls_outputs']
    pred_features = outputs['features']
    # fmt: on

    # print(cls_outputs.shape, gt_labels.shape)

    # Log prediction accuracy
    # log_accuracy(pred_class_logits, gt_labels)

    loss_dict = {}
    loss_names = cfg.MODEL.LOSSES.NAME

    if "CrossEntropyLoss" in loss_names:
        loss_dict['loss_cls'] = cross_entropy_loss(
            cls_outputs,
            gt_labels,
            cfg.MODEL.LOSSES.CE.EPSILON
        )

    if "TripletLoss" in loss_names:
        loss_dict['loss_triplet'] = triplet_loss(
            pred_features,
            gt_labels,
        )

    if "CircleLoss" in loss_names:
        loss_dict['loss_circle'] = circle_loss(
            pred_features,
            gt_labels,
        )

    return loss_dict