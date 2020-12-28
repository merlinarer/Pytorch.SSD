import os
import torch

def save_checkpoint(checkpoint, epoch, cfg):
    save_filename = 'epoch_%s.pth' % epoch
    save_path = os.path.join('{}'.format(cfg.OUTPUT_DIR), save_filename)
    torch.save(checkpoint, save_path)