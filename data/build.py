# encoding: utf-8
"""
@author:  merlin
@contact: merlinarer@gmail.com
"""
import torch
from utils.collate import collate
from torch.utils.data import DataLoader
from data.dataloader import VOCDataset, COCODataset
from data import SSDAugmentation, BaseTransform


def build_dataloader(cfg):
    assert cfg.DATA.NAME in ['VOC', 'COCO'], 'Only VOC and COCO is supported !'
    if cfg.DATA.NAME == 'VOC':
        train_dataset = VOCDataset(root=cfg.DATA.TRAINROOT,
                                   transform=SSDAugmentation(),
                                   type='trainval')
        val_dataset = VOCDataset(root=cfg.DATA.VALROOT,
                                 transform=BaseTransform(),
                                 type='test')
    else:
        train_dataset = COCODataset(root=cfg.DATA.TRAINROOT,
                                    transform=SSDAugmentation(),
                                    type='train2017')
        val_dataset = COCODataset(root=cfg.DATA.VALROOT,
                                  transform=BaseTransform(),
                                  type='val2017')

    if cfg.DISTRIBUTE:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset,
                                  batch_size=cfg.SOLVER.BATCH_SIZE,
                                  num_workers=8,
                                  collate_fn=collate,
                                  sampler=sampler,
                                  pin_memory=False)
    else:
        train_loader = DataLoader(train_dataset,
                                  batch_size=cfg.SOLVER.BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=8,
                                  collate_fn=collate,
                                  pin_memory=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.SOLVER.VAL_BATCH_SIZE,
                            shuffle=True,
                            num_workers=8,
                            collate_fn=collate,
                            pin_memory=False)

    return train_loader, val_loader
