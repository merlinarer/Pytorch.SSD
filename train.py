# encoding: utf-8
"""
@author:  merlin
@contact: merlinarer@gmail.com
"""

import argparse
import os
import torch
from torch import nn
from torch.autograd import Variable
import time
from torchvision import transforms
from torch.backends import cudnn
from torch.utils.data import DataLoader
import logging
import numpy as np

from config import cfg
from data.dataloader import MyVOCDataset
from utils.logger import setup_logger
from utils.checkpoint import save_checkpoint, load_model, resume_from
from utils.lr import adjust_learning_rate
from utils.collate import train_collate, val_collate

from modeling.build import SSD
from modeling.losses import ssdloss
from data import SSDAugmentation, BaseTransform
import torch.optim as optim


def train(cfg):
    # logger
    logger = logging.getLogger(name="merlin.baseline.train")
    logger.info("training...")

    # prepare dataset
    train_dataset = MyVOCDataset(root=cfg.DATA.ROOT,
                                 transform=SSDAugmentation(),
                                 type='trainval')
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.SOLVER.BATCH_SIZE,
                              shuffle=True,
                              num_workers=8,
                              collate_fn=train_collate,
                              pin_memory=False)
    val_dataset = MyVOCDataset(root=cfg.DATA.VALROOT + 'VOC2007',
                               transform=BaseTransform(),
                               type='test')
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.TEST.IMS_PER_BATCH,
                            shuffle=True,
                            num_workers=8,
                            collate_fn=val_collate,
                            pin_memory=False)
    num_classes = cfg.MODEL.HEADS.NUM_CLASSES

    # prepare model
    variance = [0.1, 0.2]
    model = SSD(num_classes=num_classes, variance=variance)

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=cfg.SOLVER.BASE_LR,
                              momentum=cfg.SOLVER.MOMENTUM,
                              weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.SOLVER.BASE_LR,
                               weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        raise Exception('Only Adam and SGD is supported !')

    if cfg.RESUME:
        model_state_dict, optimizer_state_dict, epoch = resume_from(cfg.LOAD_FROM)
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        start_epoch = epoch
        sum_it = start_epoch * cfg.SOLVER.BATCH_SIZE
    else:
        model.init_weights(pretrained=cfg.MODEL.BACKBONE.PRETRAIN_PATH)
        start_epoch = 0
        sum_it = 0

    model = model.cuda()
    model = nn.DataParallel(model)

    loss_fn = ssdloss.SSDLoss(variances=variance)

    # Train and val
    since = time.time()
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        model.train(True)
        logger.info("Epoch {}/{}".format(epoch, cfg.SOLVER.MAX_EPOCHS - 1))
        logger.info('-' * 10)

        running_loss = 0.0
        # Iterate over data
        it = 0
        for imgs, gt_bb, gt_label in train_loader:
            it += 1
            sum_it += 1
            inputs = Variable(imgs.cuda(), requires_grad=False)
            now_batch_size, c, h, w = imgs.shape
            if now_batch_size < cfg.SOLVER.BATCH_SIZE:  # skip the last batch
                continue

            # zero the gradients
            optimizer.zero_grad()

            # forward
            pred_bb, pred_label, anchor = model(inputs)

            with torch.no_grad():
                gt_bb = [Variable(ann.cuda(), requires_grad=False) for ann in gt_bb]
                gt_label = [Variable(ann.cuda(), requires_grad=False) for ann in gt_label]

            loss = loss_fn(pred_bb, pred_label, gt_bb, gt_label, anchor)
            loss.backward()
            optimizer.step()
            adjust_learning_rate(cfg, optimizer, cfg.SOLVER.GAMMA, sum_it)

            # statistics
            with torch.no_grad():
                running_loss += loss

            if it % 10 == 0:
                logger.info(
                    'epoch {}, iter {}, sum_it: {}, loss: {:.3f}, lr: {:.5f}'.format(
                        epoch, it, sum_it, running_loss / it,
                        optimizer.param_groups[0]['lr']))

        epoch_loss = running_loss / it
        logger.info('epoch {} loss: {:.4f}'.format(epoch, epoch_loss))

        # save checkpoint
        if (epoch + 1) % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            checkpoint = {'epoch': epoch + 1,
                          'model': model.module.state_dict() if (len(
                              cfg.MODEL.DEVICE_ID) - 2) > 1 else model.state_dict(),
                          'optimizer': optimizer.state_dict()
                          }
            save_checkpoint(checkpoint, epoch, cfg)

        time_elapsed = time.time() - since
        logger.info('Training complete in {:.0f}m {:.0f}s\n'.format(
            time_elapsed // 60, time_elapsed % 60))

        # evaluate
        if (epoch + 1) % cfg.SOLVER.EVAL_PERIOD == 0:
            logger.info('evaluate...')
            model.train(False)
            model.module.head.nms = True

            from eval import VOCMap
            ap = VOCMap(cfg.DATA.VALROOT,
                        21, logger=logger, num_images=len(val_dataset))
            ap(model, val_loader)
            model.module.head.nms = False


def main():
    parser = argparse.ArgumentParser(description="Merlin Baseline Training")
    parser.add_argument("--config_file",
                        default="",
                        help="path to config file",
                        type=str)
    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger(name="merlin.baseline", output=output_dir)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True
    train(cfg)


if __name__ == '__main__':
    main()
