import argparse
import torch

class BaseOptions:

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.gpu_num = torch.cuda.device_count()
        self.initialize()
        
    def initialize(self):

        # Experiment id
        self.parser.add_argument('--run_id', type=str, required=True) 
        self.parser.add_argument('--gpu_id', type=int, default=0) 
        self.parser.add_argument('--ckpt_id', type=str, default=None)
        self.parser.add_argument('--model_id', type=str, default="hififace")
            
        self.parser.add_argument('--valid_dataset_root', type=str, \
            default='assets/valid', help="dir path or None")

        # Log
        self.parser.add_argument('--loss_cycle', type=str, default=10)
        self.parser.add_argument('--test_cycle', type=str, default=1000)
        self.parser.add_argument('--ckpt_cycle', type=str, default=10000)
        self.parser.add_argument('--save_root', type=str, default="train_result")

        # Loss Weight pool
        # W_id, W_shape, W_adv, W_recon, W_seg, W_cycle, 
        # W_lpips, W_attr, W_fm, W_lm, W_face3d

        # Multi GPU
        self.parser.add_argument('--gpu_num', default=self.gpu_num)
        self.parser.add_argument('--isMaster', default=True)
        self.parser.add_argument('--use_mGPU', action='store_true')

        # Use wandb
        self.parser.add_argument('--use_wandb', action='store_true')

    def parse(self):
        args = self.parser.parse_args()
        return args
