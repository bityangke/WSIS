# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
This script is a simplified version of the training script in detectron2/tools.
"""

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.utils.logger import setup_logger

from wsis.config import add_wsis_config
from wsis.train_loop import *


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_wsis_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)  # modify the args by using command line
    cfg.freeze()
    default_setup(cfg, args)
    # reproducibility
    # torch.backends.cudnn.deterministic = True  # this will hurt the speed

    # Setup logger for "WSIS" sub module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="wsis")
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = eval(args.train_loop).build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )  # resume from the latest checkpoint, or load pre-trained parameters
        res = eval(args.train_loop).test(cfg, model)
        # if comm.is_main_process():
        #     verify_results(cfg, res)
        return res

    trainer = eval(args.train_loop)(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--train_loop", default="WSTrainer", type=str,
        help="What kind of training protocal: Trainer, SemiTrainer, MetaSemiTrainer, MRoundSemiTrainer"
    )
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
