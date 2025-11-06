import os
import random
import argparse
import numpy as np
import torch

from utils.config import _C as cfg
from utils.logger import setup_logger

from trainer import Trainer


def setup_cfg(args):
    """
    args를 기반으로 config 파일을 로드하고 병합하여 cfg 객체를 반환합니다.
    """
    cfg_data_file = os.path.join("./configs/data", args.data + ".yaml")
    cfg_backbone_file = os.path.join("./configs/backbone", args.backbone + ".yaml")
    cfg_method_file = os.path.join("./configs/method", args.method + ".yaml")

    cfg.defrost()
    cfg.merge_from_file(cfg_data_file)
    cfg.merge_from_file(cfg_backbone_file)
    cfg.merge_from_file(cfg_method_file)
    cfg.merge_from_list(args.opts)
    
    # 이 함수는 수정된 cfg 객체를 반환할 필요는 없지만, 명시적으로 반환하는 것이 좋은 습관입니다.
    # _C as cfg로 import 했기 때문에, 전역적으로 cfg가 수정됩니다.
    return cfg

def main(args):

    setup_cfg(args)

    """
    cfg_data_file = os.path.join("./configs/data", args.data + ".yaml")
    cfg_backbone_file = os.path.join("./configs/backbone", args.backbone + ".yaml")
    cfg_method_file = os.path.join("./configs/method", args.method + ".yaml")

    cfg.defrost()
    cfg.merge_from_file(cfg_data_file)
    cfg.merge_from_file(cfg_backbone_file)
    cfg.merge_from_file(cfg_method_file)
    cfg.merge_from_list(args.opts)
    """

    print("======================================================")
    print("!!! FINAL BACKBONE CONFIG BEING USED:", cfg.backbone)
    print("======================================================")

    if args.weights_path:
        # yacs cfg 객체를 수정 가능하도록 잠시 defrost
        cfg.defrost()
        # argparse로 받은 값을 cfg에 대문자 WEIGHTS_PATH로 업데이트
        cfg.WEIGHTS_PATH = args.weights_path
    # cfg.freeze()

    if cfg.output_dir is None:
        cfg_name = "_".join([args.data, args.backbone, args.method])
        opts_name = "".join(["_" + item for item in args.opts])
        cfg.output_dir = os.path.join("./output", cfg_name + opts_name)
    else:
        cfg.output_dir = os.path.join("./output", cfg.output_dir)
    print("Output directory: {}".format(cfg.output_dir))
    setup_logger(cfg.output_dir)
    
    print("** Config **")
    print(cfg)
    print("************")
    
    if cfg.seed is not None:
        seed = cfg.seed
        print("Setting fixed seed: {}".format(seed))
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if cfg.deterministic:
        print("Setting deterministic operations.")
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        # Turn off filling uninitialized memory for better efficiency if not use uninitialized memory.
        # https://pytorch.org/docs/stable/notes/randomness.html#filling-uninitialized-memory
        torch.utils.deterministic.fill_uninitialized_memory = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    trainer = Trainer(cfg)
    
    if cfg.model_dir is not None:
        trainer.load_model(cfg.model_dir)

    if cfg.test_only == True:
        if cfg.model_dir is None:
            cfg.model_dir = cfg.output_dir[:cfg.output_dir.index("_test_only_True")]
            print("Model directory: {}".format(cfg.model_dir))
        
        trainer.load_model(cfg.model_dir)
        trainer.test()
        return
    
    if cfg.zero_shot:
        trainer.test()
        return
    
    
    trainer.train()
    trainer.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="", help="data config file")
    parser.add_argument("--backbone", "-b", type=str, default="", help="backbone config file")
    parser.add_argument("--method", "-m", type=str, default="", help="fine-tuning method config file")
    parser.add_argument("--weights-path", type=str, default=None,
                        help="Path to the pre-computed classifier weights pt file.")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="modify config options using the command-line")
    args = parser.parse_args()
    main(args)