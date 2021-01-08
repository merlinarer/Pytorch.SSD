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


def build_dataloader(datasetname,
                     trainroot,
                     valroot,
                     batchsize,
                     test_bs,
                     distribute=True):
    assert datasetname in ['VOC', 'COCO'], 'Only VOC and COCO is supported !'
    if datasetname == 'VOC':
        train_dataset = VOCDataset(root=trainroot,
                                   transform=SSDAugmentation(),
                                   type='trainval')
        val_dataset = VOCDataset(root=valroot,
                                 transform=BaseTransform(),
                                 type='test')
    else:
        train_dataset = COCODataset(root=trainroot,
                                    transform=SSDAugmentation(),
                                    type='train2017')
        val_dataset = COCODataset(root=valroot,
                                  transform=BaseTransform(),
                                  type='val2017')

    if distribute:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset,
                                  batch_size=batchsize,
                                  num_workers=0,
                                  collate_fn=collate,
                                  sampler=sampler,
                                  pin_memory=False)
    else:
        train_loader = DataLoader(train_dataset,
                                  batch_size=batchsize,
                                  shuffle=True,
                                  num_workers=8,
                                  collate_fn=collate,
                                  pin_memory=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=test_bs,
                            shuffle=True,
                            num_workers=8,
                            collate_fn=collate,
                            pin_memory=False)

    return train_loader, val_loader
