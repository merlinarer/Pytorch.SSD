# encoding: utf-8
"""
@author:  merlin
@contact: merlinarer@gmail.com
"""

import argparse
import os
import torch
from torch.backends import cudnn

import importlib
from utils.logger import setup_logger
from modeling import build_detector
from utils.optimizer import build_optimizer
import torch.multiprocessing as mp
from utils.trainer import train_with_ddp, train_with_dp


def main():
    parser = argparse.ArgumentParser(description="Merlin Baseline Training")
    parser.add_argument("--config_file",
                        default="",
                        help="path to config file",
                        type=str)
    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    assert args.config_file != "", 'Config_file is in need !'
    cfgname = os.path.basename(args.config_file).split('.')[0]
    cfg = importlib.import_module('config.{}_defaults'.format(cfgname)).Cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger(name="evig.detection", output=output_dir)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cudnn.benchmark = True
    nprocs = torch.cuda.device_count()

    # build model / optimizer
    model = build_detector(cfg)
    optimizer = build_optimizer(cfg.SOLVER.OPTIMIZER_NAME,
                                model,
                                cfg.SOLVER.BASE_LR,
                                cfg.SOLVER.MOMENTUM,
                                cfg.SOLVER.WEIGHT_DECAY)

    if cfg.DISTRIBUTE:
        mp.spawn(train_with_ddp, nprocs=nprocs, join=True,
                 args=(nprocs, cfg, model, optimizer))
    else:
        train_with_dp(cfg, model, optimizer)


if __name__ == '__main__':
    main()
