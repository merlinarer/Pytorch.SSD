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
from solver.optimizer import make_optimizer
from solver.lr_scheduler import WarmupMultiStepLR

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


# def detection_collate(batch):
#     imgs = []
#     boxs = []
#     labels = []
#     max_num_annots = max(sample[3] for sample in batch)
#     boxs_padded = torch.ones((max_num_annots.item(), 4), dtype=torch.float32) * -1
#     labels_padded = torch.ones((max_num_annots.item(), 1), dtype=torch.int8) * -1
#     for sample in batch:
#         imgs.append(sample[0])
#         # boxs
#         len_sample = sample[1].shape[0]
#         boxs_padded[0:len_sample, :] = torch.FloatTensor(sample[1])
#         boxs.append(boxs_padded)
#         # labels
#         len_sample = sample[2].shape[0]
#         labels_padded[0:len_sample, :] = torch.Tensor(sample[2][:, np.newaxis])
#         labels.append(labels_padded)
#     return torch.stack(imgs, 0), torch.stack(boxs, 0), torch.stack(labels, 0)


def train(cfg):
    # logger
    logger = logging.getLogger(name="merlin.baseline.train")
    logger.info("training...")

    # prepare dataset
    train_dataset = MyVOCDataset(root=cfg.DATA.ROOT, transform=SSDAugmentation(), type='train')
    val_dataset = MyVOCDataset(root=cfg.DATA.ROOT, transform=SSDAugmentation(), type='val')
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.SOLVER.BATCH_SIZE,
                              shuffle=True,
                              num_workers=8,
                              collate_fn=detection_collate,
                              pin_memory=False)
    val_loader = DataLoader(val_dataset,
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

    # prepare solver
    # optimizer = make_optimizer(cfg, model)
    # scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
    #                               cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9,
                          weight_decay=5e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.1)

    start_epoch = 0
    loss_fn = ssdloss.SSDLoss()

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
            inputs = Variable(imgs.cuda(), requires_grad=False)
            now_batch_size, c, h, w = imgs.shape
            if now_batch_size < cfg.SOLVER.BATCH_SIZE:  # skip the last batch
                continue

            # zero the gradients
            optimizer.zero_grad()

            # forward
            pred_bb, pred_label = model(inputs)

            with torch.no_grad():
                gt_bb = [Variable(ann.cuda(), requires_grad=False) for ann in gt_bb]
                gt_label = [Variable(ann.cuda(), requires_grad=False) for ann in gt_label]

            loss = loss_fn(pred_bb, pred_label, gt_bb, gt_label)

            loss.backward()
            optimizer.step()

            # statistics
            with torch.no_grad():
                running_loss += loss

            if it % 10 == 0:
                logger.info(
                    'epoch {}, iter {}, loss: {:.3f}, lr: {:.5f}'.format(
                        epoch, it, running_loss / it,
                        optimizer.param_groups[0]['lr']))

        epoch_loss = running_loss / it
        # if epoch in cfg.SOLVER.STEPS:
        #     scheduler.step()
        logger.info('epoch {} loss: {:.4f}'.format(epoch, epoch_loss))

        # save checkpoint
        if epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            checkpoint = {'epoch': epoch + 1,
                          'model': model.module.state_dict() if (len(
                              cfg.MODEL.DEVICE_ID) - 2) > 1 else model.state_dict(),
                          'optimizer': optimizer.state_dict()
                          }
            save_checkpoint(checkpoint, epoch, cfg)

        # evaluate
        # if epoch % cfg.SOLVER.EVAL_PERIOD == 0:
        #     logger.info('evaluate...')
        #     model.train(False)
        #
        #     total = 0.0
        #     correct = 0.0
        #     for data in val_loader:
        #         inputs, labels = data
        #         inputs = Variable(inputs.cuda().detach())
        #         labels = Variable(labels.cuda().detach())
        #         with torch.no_grad():
        #             out = model(inputs)
        #             _, preds = torch.max(out['pred_class_logits'], 1)
        #             c = (preds == labels).squeeze()
        #             total += c.size(0)
        #             correct += c.float().sum().item()
        #     acc = correct / total
        #     logger.info('eval acc:{:.4f}'.format(acc))

        time_elapsed = time.time() - since
        logger.info('Training complete in {:.0f}m {:.0f}s\n'.format(
            time_elapsed // 60, time_elapsed % 60))

    return model


def main():
    parser = argparse.ArgumentParser(description="Merlin Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
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
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # logger.info("Using {} GPUS".format(num_gpus))
    train(cfg)


if __name__ == '__main__':
    main()
