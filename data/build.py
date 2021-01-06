from utils.collate import train_collate, val_collate
from torch.utils.data import DataLoader
from data.dataloader import MyVOCDataset
from data import SSDAugmentation, BaseTransform


def build_dataloader(datasetname,
                     trainroot,
                     valroot,
                     batchsize):
    assert datasetname in ['VOC', 'COCO'], 'Only VOC and COCO is supported !'
    train_dataset = MyVOCDataset(root=trainroot,
                                 transform=SSDAugmentation(),
                                 type='trainval')
    train_loader = DataLoader(train_dataset,
                              batch_size=batchsize,
                              shuffle=True,
                              num_workers=8,
                              collate_fn=train_collate,
                              pin_memory=False)
    val_dataset = MyVOCDataset(root=valroot,
                               transform=BaseTransform(),
                               type='test')
    val_loader = DataLoader(val_dataset,
                            batch_size=batchsize,
                            shuffle=True,
                            num_workers=8,
                            collate_fn=val_collate,
                            pin_memory=False)

    return train_loader, val_loader
