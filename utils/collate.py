import torch


def collate(batch):
    imgs, boxs, labels, info = [], [], [], []
    for sample in batch:
        imgs.append(sample[0])
        boxs.append(torch.FloatTensor(sample[1]))
        labels.append(torch.LongTensor(sample[2]))
    return torch.stack(imgs, 0), boxs, labels


def val_collate(batch):
    imgs = []
    boxs = []
    labels = []
    imgname = []
    h = []
    w = []
    for sample in batch:
        imgs.append(sample[0])
        boxs.append(torch.FloatTensor(sample[1]))
        labels.append(torch.LongTensor(sample[2]))
        h.append(sample[3])
        w.append(sample[4])
        imgname.append(sample[5])
    return torch.stack(imgs, 0), boxs, labels, h, w, imgname

