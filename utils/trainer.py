# encoding: utf-8
"""
@author:  merlin
@contact: merlinarer@gmail.com
"""

import torch
from torch import nn
from torch.autograd import Variable
import time
import logging

from utils.logger import setup_logger
from utils.checkpoint import save_checkpoint, resume_from
from utils.lr import adjust_learning_rate

from data import build_dataloader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.eval_voc import VOCMap
from utils.eval_coco import COCOMap


def train_with_ddp(local_rank,
                   nprocs,
                   cfg,
                   model,
                   optimizer):
    # logger
    logger = setup_logger(name="evig.detection",
                          distributed_rank=local_rank,
                          output=cfg.OUTPUT_DIR)
    logger.info("training...")

    dist.init_process_group(backend='nccl',
                            init_method='tcp://0.0.0.0:8891',
                            world_size=nprocs,
                            rank=local_rank)

    train_loader, val_loader = build_dataloader(cfg.DATA.NAME,
                                                cfg.DATA.TRAINROOT,
                                                cfg.DATA.VALROOT,
                                                cfg.SOLVER.BATCH_SIZE,
                                                cfg.SOLVER.VAL_BATCH_SIZE)

    if cfg.RESUME:
        model_state_dict, optimizer_state_dict, epoch = resume_from(cfg.LOAD_FROM)
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        start_epoch = epoch
    else:
        start_epoch = 0

    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # Train and val
    since = time.time()
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        model.train(True)
        logger.info("Epoch {}/{}".format(epoch, cfg.SOLVER.MAX_EPOCHS - 1))
        logger.info('-' * 10)

        running_loss = 0.0
        # Iterate over data
        it = 0
        for imgs, gt_bb, gt_label, _ in train_loader:
            it += 1
            inputs = Variable(imgs.cuda(), requires_grad=False)
            now_batch_size, c, h, w = imgs.shape
            if now_batch_size < cfg.SOLVER.BATCH_SIZE:  # skip the last batch
                continue

            # zero the gradients
            optimizer.zero_grad()

            # forward
            out = model(inputs)

            with torch.no_grad():
                gt_bb = [Variable(ann.cuda(), requires_grad=False) for ann in gt_bb]
                gt_label = [Variable(ann.cuda(), requires_grad=False) for ann in gt_label]

            loss = model.module.loss_fn(out, (gt_bb, gt_label))
            loss.backward()
            optimizer.step()
            past_iter = start_epoch * len(train_loader.dataset) / cfg.SOLVER.BATCH_SIZE
            adjust_learning_rate(cfg, optimizer, cfg.SOLVER.GAMMA, past_iter + it)

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
        if local_rank == 0 and (epoch + 1) % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            checkpoint = {'epoch': epoch + 1,
                          'model': model.module.state_dict() if (len(
                              cfg.MODEL.DEVICE_ID) - 2) > 1 else model.state_dict(),
                          'optimizer': optimizer.state_dict()
                          }
            save_checkpoint(checkpoint, epoch + 1, cfg)

        time_elapsed = time.time() - since
        logger.info('Training complete in {:.0f}m {:.0f}s\n'.format(
            time_elapsed // 60, time_elapsed % 60))

        # evaluate
        if local_rank == 0 and (epoch) % cfg.SOLVER.EVAL_PERIOD == 0:
            logger.info('evaluate...')
            model.train(False)
            model.module.head.nms = True

            assert cfg.DATA.NAME in ['VOC', 'COCO'], \
                'Only VOC and COCO is supported !'
            if cfg.DATA.NAME == 'VOC':
                from utils.eval_voc import VOCMap
                ap = VOCMap(cfg.DATA.VALROOT,
                            cfg.MODEL.HEADS.NUM_CLASSES,
                            num_images=len(val_loader.dataset),
                            type='test')
                ap(model, val_loader)
            else:
                from utils.eval_coco import COCOMap
                ap = COCOMap(cfg.DATA.VALROOT,
                             num_images=len(val_loader.dataset),
                             type='val2017')
                ap(model, val_loader)

            model.module.head.nms = False


def train_with_dp(cfg,
                  model,
                  optimizer):
    # logger
    logger = logging.getLogger(name="evig.detection")
    logger.info("training...")

    train_loader, val_loader = build_dataloader(cfg.DATA.NAME,
                                                cfg.DATA.TRAINROOT,
                                                cfg.DATA.VALROOT,
                                                cfg.SOLVER.BATCH_SIZE,
                                                cfg.SOLVER.VAL_BATCH_SIZE,
                                                distribute=False)

    if cfg.RESUME:
        model_state_dict, optimizer_state_dict, epoch = resume_from(cfg.LOAD_FROM)
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        start_epoch = epoch
    else:
        start_epoch = 0

    model = model.cuda()
    model = nn.DataParallel(model)

    # Train and val
    since = time.time()
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        model.train(True)
        logger.info("Epoch {}/{}".format(epoch, cfg.SOLVER.MAX_EPOCHS - 1))
        logger.info('-' * 10)

        running_loss = 0.0
        # Iterate over data
        it = 0
        for imgs, gt_bb, gt_label, _ in train_loader:
            it += 1
            inputs = Variable(imgs.cuda(), requires_grad=False)
            now_batch_size, c, h, w = imgs.shape
            if now_batch_size < cfg.SOLVER.BATCH_SIZE:  # skip the last batch
                continue

            # zero the gradients
            optimizer.zero_grad()

            # forward
            out = model(inputs)

            with torch.no_grad():
                gt_bb = [Variable(ann.cuda(), requires_grad=False) for ann in gt_bb]
                gt_label = [Variable(ann.cuda(), requires_grad=False) for ann in gt_label]

            loss = model.module.loss_fn(out, (gt_bb, gt_label))
            loss.backward()
            optimizer.step()
            past_iter = start_epoch * len(train_loader.dataset) / cfg.SOLVER.BATCH_SIZE
            adjust_learning_rate(cfg, optimizer, cfg.SOLVER.GAMMA, past_iter + it)

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
        if (epoch) % cfg.SOLVER.EVAL_PERIOD == 0:
            logger.info('evaluate...')
            model.train(False)
            model.module.head.nms = True

            assert cfg.DATA.NAME in ['VOC', 'COCO'], \
                'Only VOC and COCO is supported !'
            if cfg.DATA.NAME == 'VOC':
                ap = VOCMap(cfg.DATA.VALROOT,
                            cfg.MODEL.HEADS.NUM_CLASSES,
                            num_images=len(val_loader.dataset),
                            type='test')
                ap(model, val_loader)
            else:
                ap = COCOMap(cfg.DATA.VALROOT,
                             num_images=len(val_loader.dataset),
                             type='val2017')
                ap(model, val_loader)

            model.module.head.nms = False
