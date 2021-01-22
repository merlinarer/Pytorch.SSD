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
from torch.utils.tensorboard import SummaryWriter
from utils.eval_voc import VOCMap
from utils.eval_coco import COCOMap
from utils.optimizer import build_optimizer
from modeling import build_detector


def train_with_ddp(local_rank,
                   nprocs,
                   cfg,
                   logger=None):
    # mpi init
    if not cfg.LAUNCH:
        dist.init_process_group(backend='nccl',
                                init_method='tcp://0.0.0.0:8891',
                                world_size=nprocs,
                                rank=local_rank)
        logger = setup_logger(name="evig.detection",
                              distributed_rank=local_rank,
                              output=cfg.OUTPUT_DIR)
    logger.info("training...")

    # set cuda device
    torch.cuda.set_device(local_rank)

    writer = SummaryWriter(log_dir='log')

    model = build_detector(cfg)
    train_loader, val_loader = build_dataloader(cfg)
    optimizer = build_optimizer(cfg, model)

    if cfg.RESUME:
        model_state_dict, optimizer_state_dict, epoch = resume_from(cfg.LOAD_FROM)
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        for k, v in optimizer.state.items():
            if 'momentum_buffer' not in v:
                continue
            optimizer.state[k]['momentum_buffer'] = \
                optimizer.state[k]['momentum_buffer'].cuda()
        start_epoch = epoch
    else:
        start_epoch = 0

    model.cuda(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    assert cfg.DATA.NAME in ['VOC', 'COCO'], \
        'Only VOC and COCO is supported !'
    if cfg.DATA.NAME == 'VOC':
        ap = VOCMap(cfg.DATA.VALROOT,
                    cfg.MODEL.HEADS.NUM_CLASSES,
                    num_images=len(val_loader.dataset),
                    type='test')
    else:
        ap = COCOMap(cfg.DATA.VALROOT,
                     num_images=len(val_loader.dataset),
                     type='val2017')

    # Train and val
    since = time.time()
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        model.train(True)
        train_loader.sampler.set_epoch(epoch)
        logger.info("Epoch {}/{}".format(epoch + 1, cfg.SOLVER.MAX_EPOCHS))
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
            past_iter = int(epoch * len(train_loader.dataset) / cfg.SOLVER.BATCH_SIZE)
            adjust_learning_rate(cfg, optimizer, past_iter + it)

            loss_value = loss.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
            # statistics
            with torch.no_grad():
                running_loss += loss_value.detach().item()

            if it % 10 == 0 and dist.get_rank()==0:
                logger.info(
                    'epoch {}, iter {}, loss: {:.3f}, lr: {:.5f}'.format(
                        epoch + 1, it, running_loss / it,
                        optimizer.param_groups[0]['lr']))
                writer.add_scalar('loss', running_loss / it, global_step=past_iter + it)

        epoch_loss = running_loss / it
        logger.info('epoch {} loss: {:.4f}'.format(epoch + 1, epoch_loss))

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
        if local_rank == 0 and (epoch + 1) % cfg.SOLVER.EVAL_PERIOD == 0:
            logger.info('evaluate...')
            model.eval()
            model.module.head.nms = True
            ap(model, val_loader)
            model.module.head.nms = False


def train_with_dp(cfg):
    # logger
    logger = logging.getLogger(name="evig.detection")
    logger.info("training...")

    model = build_detector(cfg)
    train_loader, val_loader = build_dataloader(cfg)
    optimizer = build_optimizer(cfg, model)

    if cfg.RESUME:
        model_state_dict, optimizer_state_dict, epoch = resume_from(cfg.LOAD_FROM)
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        for k, v in optimizer.state.items():
            if 'momentum_buffer' not in v:
                continue
            optimizer.state[k]['momentum_buffer'] = \
                optimizer.state[k]['momentum_buffer'].cuda()
        start_epoch = epoch
    else:
        start_epoch = 0

    model = model.cuda()
    model = nn.DataParallel(model)

    assert cfg.DATA.NAME in ['VOC', 'COCO'], \
        'Only VOC and COCO is supported !'
    if cfg.DATA.NAME == 'VOC':
        ap = VOCMap(cfg.DATA.VALROOT,
                    cfg.MODEL.HEADS.NUM_CLASSES,
                    num_images=len(val_loader.dataset),
                    type='test')
    else:
        ap = COCOMap(cfg.DATA.VALROOT,
                     num_images=len(val_loader.dataset),
                     type='val2017')

    # Train and val
    since = time.time()
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        model.train(True)
        logger.info("Epoch {}/{}".format(epoch + 1, cfg.SOLVER.MAX_EPOCHS))
        logger.info('-' * 10)

        running_loss = 0.0
        # Iterate over data
        it = 0
        for imgs, gt_bb, gt_label, _ in train_loader:
            inputs = Variable(imgs.cuda(), requires_grad=False)
            now_batch_size, c, h, w = imgs.shape
            if now_batch_size < cfg.SOLVER.BATCH_SIZE:  # skip the last batch
                continue
            it += 1

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
            past_iter = int(epoch * len(train_loader.dataset) / cfg.SOLVER.BATCH_SIZE)
            adjust_learning_rate(cfg, optimizer, past_iter + it)

            # statistics
            with torch.no_grad():
                running_loss += loss

            if it % 10 == 0:
                logger.info(
                    'epoch {}, iter {}, loss: {:.3f}, lr: {:.5f}'.format(
                        epoch + 1, it, running_loss / it,
                        optimizer.param_groups[0]['lr']))

        epoch_loss = running_loss / it
        logger.info('epoch {} loss: {:.4f}'.format(epoch + 1, epoch_loss))

        # save checkpoint
        if (epoch + 1) % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
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
        if (epoch + 1) % cfg.SOLVER.EVAL_PERIOD == 0:
            logger.info('evaluate...')
            model.train(False)
            model.module.head.nms = True
            ap(model, val_loader)
            model.module.head.nms = False
