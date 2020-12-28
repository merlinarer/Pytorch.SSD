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
from utils.checkpoint import save_checkpoint

from modeling.build import SSD
from modeling.losses import ssdloss
from data import SSDAugmentation
from torch.optim.lr_scheduler import ExponentialLR
import torch.optim as optim


def detection_collate(batch):
    imgs = []
    boxs = []
    labels = []
    for sample in batch:
        imgs.append(sample[0])
        boxs.append(torch.FloatTensor(sample[1]))
        labels.append(torch.FloatTensor(sample[2]))
    return torch.stack(imgs, 0), boxs, labels


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
                              collate_fn=detection_collate,
                              pin_memory=False)
    num_classes = cfg.MODEL.HEADS.NUM_CLASSES

    # prepare model
    model = SSD(num_classes=num_classes)
    model.init_weights(pretrained=cfg.MODEL.BACKBONE.PRETRAIN_PATH)
    model = model.cuda()
    model = nn.DataParallel(model)

    optimizer = optim.SGD(model.parameters(),
                          lr=cfg.SOLVER.BASE_LR,
                          momentum=cfg.SOLVER.WEIGHT_DECAY,
                          weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    scheduler = ExponentialLR(optimizer, gamma=cfg.SOLVER.GAMMA)

    loss_fn = ssdloss.SSDLoss()

    # Train and val
    since = time.time()
    sum_it = 0
    start_epoch = 0
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
            if sum_it in cfg.SOLVER.STEPS:
                scheduler.step()
                logger.info('sum_iter {}'.format(sum_it))

            # statistics
            with torch.no_grad():
                running_loss += loss

            if it % 10 == 0:
                logger.info(
                    'epoch {}, iter {}, loss: {:.3f}, lr: {:.5f}'.format(
                        epoch, it, running_loss / it,
                        optimizer.param_groups[0]['lr']))

        epoch_loss = running_loss / it
        logger.info('epoch {} loss: {:.4f}'.format(epoch, epoch_loss))

        # save checkpoint
        if epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            checkpoint = {'epoch': epoch + 1,
                          'model': model.module.state_dict() if (len(
                              cfg.MODEL.DEVICE_ID) - 2) > 1 else model.state_dict(),
                          'optimizer': optimizer.state_dict()
                          }
            save_checkpoint(checkpoint, epoch, cfg)

        time_elapsed = time.time() - since
        logger.info('Training complete in {:.0f}m {:.0f}s\n'.format(
            time_elapsed // 60, time_elapsed % 60))

    return model


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
