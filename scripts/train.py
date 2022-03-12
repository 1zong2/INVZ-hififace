import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("./submodel/")
import torch
import wandb
import os
from lib.options import BaseOptions
from lib.model_loader import CreateModel



def train(gpu, args): 
    torch.cuda.set_device(gpu)
    model, args, step = CreateModel(gpu, args)

    # Initialize wandb to gather and display loss on dashboard 
    if args.isMaster and args.use_wandb:
        wandb.init(project=args.model_id, name=args.run_id)

    # Training loop
    global_step = step if step else 0
    while global_step < args.max_step:
        result = model.train_step()

        if args.isMaster:
            # Save and print loss
            if global_step % args.loss_cycle == 0:
                if args.use_wandb:
                    wandb.log(model.loss_collector.loss_dict)
                model.loss_collector.print_loss(global_step)

            # Save image
            if global_step % args.test_cycle == 0:
                model.save_image(result, global_step)

                if args.valid_dataset_root:
                    model.validation(global_step) 

            # Save checkpoint parameters 
            if global_step % args.ckpt_cycle == 0:
                model.save_checkpoint(global_step)

        global_step += 1


if __name__ == "__main__":
    args = BaseOptions().parse()
    os.makedirs(args.save_root, exist_ok=True)

    # Set up multi-GPU training
    if args.use_mGPU:  
        torch.multiprocessing.spawn(train, nprocs=args.gpu_num, args=(args, ))

    # Set up single GPU training
    else:
        train(args.gpu_id, args)
