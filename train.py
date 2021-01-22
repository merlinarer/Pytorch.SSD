# encoding: utf-8
"""
@author:  merlin
@contact: merlinarer@gmail.com
"""

import argparse
import os
import torch
from torch.backends import cudnn
import torch.distributed as dist
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
    parser.add_argument('--local_rank', help='local rank', type=int, default=-1)
    parser.add_argument("--local_world_size", type=int, default=1)
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
    if cfg.DISTRIBUTE and cfg.LAUNCH:
        env_dict = {
            key: os.environ[key]
            for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
        }

        print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
        dist.init_process_group(backend="nccl")
        print(
            f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "
            + f"rank = {dist.get_rank()}, backend={dist.get_backend()} \n", end=''
        )
        local_rank = torch.distributed.get_rank()
        logger = setup_logger(name="evig.detection", output=output_dir,distributed_rank=local_rank)
    else:
        logger = setup_logger(name="evig.detection", output=output_dir,)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cudnn.benchmark = True
    nprocs = torch.cuda.device_count()
    if cfg.DISTRIBUTE:
        if cfg.LAUNCH is False:
            mp.spawn(train_with_ddp, nprocs=nprocs, join=True,
                     args=(nprocs, cfg, args))
        else:
            train_with_ddp(local_rank, None, cfg, args, logger=logger)
    else:
        train_with_dp(cfg)


if __name__ == '__main__':
    main()
