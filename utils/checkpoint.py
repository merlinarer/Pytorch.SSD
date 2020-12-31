import os
import torch


def save_checkpoint(checkpoint, epoch, cfg):
    save_filename = 'epoch_%s.pth' % epoch
    save_path = os.path.join('{}'.format(cfg.OUTPUT_DIR), save_filename)
    torch.save(checkpoint, save_path)


# def load_model(model, save_path):
#     checkpoint = torch.load(save_path)
#     if 'model' in checkpoint:
#         state_dict = checkpoint['model']
#     else:
#         state_dict = checkpoint
#     if list(state_dict.keys())[0].startswith('module.'):
#         state_dict = {k[7:]: v for k, v in checkpoint['model'].items()}
#     model.load_state_dict(state_dict)
#     return model


def load_model(save_path, rm_module=True):
    checkpoint = torch.load(save_path)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    if rm_module:
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in checkpoint['model'].items()}
    return state_dict


def resume_from(save_path, rm_module=True):
    checkpoint = torch.load(save_path)
    model_state_dict = checkpoint['model']
    optimizer_state_dict = checkpoint['optimizer']
    epoch = checkpoint['epoch']
    if rm_module:
        if list(model_state_dict.keys())[0].startswith('module.'):
            model_state_dict = {k[7:]: v for k, v in checkpoint['model'].items()}
        if list(optimizer_state_dict.keys())[0].startswith('module.'):
            optimizer_state_dict = {k[7:]: v for k, v in checkpoint['optimizer'].items()}
    return model_state_dict, optimizer_state_dict, epoch
