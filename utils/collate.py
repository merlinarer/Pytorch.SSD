import torch


def collate(batch):
    imgs, boxs, labels, info = [], [], [], []
    for sample in batch:
        imgs.append(sample[0])
        boxs.append(torch.FloatTensor(sample[1]))
        labels.append(torch.LongTensor(sample[2]))
        info.append(sample[3])
    return torch.stack(imgs, 0), boxs, labels, info

