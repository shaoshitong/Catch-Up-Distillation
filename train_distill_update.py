# From https://colab.research.google.com/drive/1LouqFBIC7pnubCOl5fhnFd33-oVJao2J?usp=sharing#scrollTo=yn1KM6WQ_7Em

"""
 python train_distill_update.py  --N 16 --gpu 4,5,6,7       --dir ./runs/cifar10-predstep-1-distill-beta20/ --latent_dir ./runs/cifar10-predstep-1-distill-beta20/test_16/zs/ \
    --traj_dir ./runs/cifar10-predstep-1-distill-beta20/test_16/     --learning_rate 2e-4 --optimizer adam --batchsize 128 \
        --iterations 100000 --flow_ckpt ./runs/cifar10-predstep-1-distill-beta20/flow_model_500000_ema.pth    --config_en configs/cifar10_en.json --config_de configs/cifar10_de.json --pred_step 1

"""
import torch
import numpy as np
import torch.nn as nn
import tensorboardX
import os,copy
from models import UNetEncoder
from guided_diffusion.unet import UNetModel
import torchvision.datasets as dsets
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from utils import straightness, get_kl, convert_ddp_state_dict_to_single,LPIPS
from dataset import DatasetWithTraj
import argparse
from tqdm import tqdm
import json 
from EMA import EMA,EMAMODEL
from network_edm import SongUNet,DWTUNet
# DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

torch.manual_seed(0)

def ddp_setup(rank, world_size,arg):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{12359+int(arg.gpu[0])}"
    # Linux
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def get_args():
    parser = argparse.ArgumentParser(description='Configs')
    parser.add_argument('--gpu', type=str, help='gpu num')
    parser.add_argument('--res', type=int,default=32, help='resolution')
    parser.add_argument('--input_nc', type=int,default=3, help='input channel')
    parser.add_argument('--dir', type=str, help='Saving directory name')
    parser.add_argument('--traj_dir', type=str, help='Trajectory directory name')
    parser.add_argument('--latent_dir', type=str, help='Latent directory name')
    parser.add_argument('--weight_cur', type=float, default = 0, help='Curvature regularization weight')
    parser.add_argument('--iterations', type=int, default = 100000, help='Number of iterations')
    parser.add_argument('--batchsize', type=int, default = 256, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default = 8e-4, help='Learning rate')
    parser.add_argument('--independent', action = 'store_true',  help='Independent assumption, q(x,z) = p(x)p(z)')
    parser.add_argument('--resume', default=None,type=str, help='Resume training from a given checkpoint')
    parser.add_argument('--flow_ckpt', type=str, default = None, help='Training state path')
    parser.add_argument('--forward_ckpt', type=str, default = None, help='Training state path')
    parser.add_argument('--preforward', type=str, default = None, help='Pretrain forward state path')
    parser.add_argument('--pred_step', type=int, default = 1, help='Predict step')
    parser.add_argument('--N', type=int, default = 16, help='Number of sampling steps')
    parser.add_argument('--num_samples', type=int, default = 64, help='Number of samples to generate')
    parser.add_argument('--no_ema', action='store_true', help='use EMA or not')
    parser.add_argument('--l_weight',type=list, default=[2.,2.],nargs='+', action='append', help='List of numbers')
    parser.add_argument('--ema_after_steps', type=int, default = 1, help='Apply EMA after steps')
    parser.add_argument('--optimizer', type=str, default = 'adamw', help='adam / adamw')
    parser.add_argument('--config_en', type=str, default = None, help='Encoder config path, must be .json file')
    parser.add_argument('--config_de', type=str, default = None, help='Decoder config path, must be .json file')

    arg = parser.parse_args()
    return arg


def distill(rank,flow_model, train_loader, iterations, optimizer, data_shape,device,arg,state_dict=None,ema_flow_model=None):
    z_fixed = torch.randn(data_shape, device=device)
    use_list = [8,11,14]
    begin_num = 0
    begin_iter = 0
    if state_dict is not None:
        flow_model.load_state_dict(state_dict['flow_model'])
        if ema_flow_model is not None:
            ema_flow_model.load_state_dict(state_dict['ema_flow_model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        begin_num = state_dict['num']
        begin_iter = state_dict['iter']
    for _num in range(begin_num,3):
        print('Trajectory: ',train_loader.dataset.traj_dir_list[use_list[_num]])
        train_loader.dataset.set_traj(use_list[_num])
        if _num!=begin_num:
            begin_iter = 0
        for i in tqdm(range(begin_iter,iterations+1)):
            optimizer.zero_grad()
            try:
                x,z = next(train_iter)
            except:
                train_iter = iter(train_loader)
                x,z = next(train_iter)
            x = x.to(device)
            z = z.to(device)
            # Learn student model
            pred_v = flow_model(z, torch.ones(z.shape[0], device=device))
            pred = z - pred_v
            loss = torch.mean((pred - x)**2)
            if ema_flow_model is not None:
                ema_pred_v = ema_flow_model(z, torch.ones(z.shape[0], device=device))
                loss+=(0.1*torch.mean((pred_v - ema_pred_v)**2))
            loss.backward()
            optimizer.step()
            if ema_flow_model is not None:
                ema_flow_model.ema_step(decay_rate=0.9999,model=flow_model)
            if i % 100 == 0:
                print(f"Iteration {i}: loss {loss.item()}")
            if i % 10000 == 0:
                flow_model.eval()
                with torch.no_grad():
                    pred_v = flow_model(z_fixed, torch.ones(z.shape[0], device=device))
                    pred = z_fixed - pred_v
                    save_image(pred * 0.5 + 0.5, os.path.join(arg.dir, f"pred_{i}.jpg"))
                flow_model.train()
                if rank == 0 and i%10000==0:
                    train_state = {}
                    train_state['flow_model'] = flow_model.module.state_dict()
                    train_state['ema_flow_model'] = ema_flow_model.ema_model.module.state_dict()
                    train_state['iter'] = i
                    train_state['num'] = _num
                    train_state['optimizer'] = optimizer.state_dict()
                    torch.save(train_state, os.path.join(arg.dir, f"train_state_lastet.pth"))
        torch.save(flow_model.state_dict(), os.path.join(arg.dir, f"flow_model_distilled_{_num}.pth"))


def parse_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main(rank: int, world_size: int, arg):
    ddp_setup(rank, world_size,arg)
    device = torch.device(f"cuda:{rank}")
    input_nc = arg.input_nc
    res = arg.res
    assert arg.config_de is not None
    if not arg.independent:
        assert arg.config_en is not None
        config_en = parse_config(arg.config_en)
    config_de = parse_config(arg.config_de)
    traj_list = []
    for i in range(16):
      _path = os.path.join(arg.traj_dir,f"trajs_{i}")
      traj_list.append(_path)
    
    train_dataset = DatasetWithTraj(traj_list,latent_dir=arg.latent_dir,input_nc = input_nc)
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=arg.batchsize, num_workers=4,sampler=torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank))
    data_shape = (arg.batchsize, input_nc, res, res)
    samples_test =  torch.randn((4, input_nc, res, res), device=device)

    config_de['total_N'] = arg.N
    if config_de['unet_type'] == 'adm':
        model_class = UNetModel
    elif config_de['unet_type'] == 'songunet':
        model_class = SongUNet
    elif config_de['unet_type'] == 'dwtunet':
        model_class = DWTUNet

    assert arg.flow_ckpt is not None
    flow_model_ckpt = torch.load(arg.flow_ckpt, map_location = 'cpu')
    flow_model = model_class(**config_de)
    flow_model.load_state_dict(convert_ddp_state_dict_to_single(flow_model_ckpt))
    print("Successfully Load Checkpoint!")

    if rank == 0:
        # Print the number of parameters in the model
        print("Begin consistency model training")
        pytorch_total_params = sum(p.numel() for p in flow_model.parameters())
        # Convert to M
        pytorch_total_params = pytorch_total_params / 1000000
        print(f"Total number of the reverse parameters: {pytorch_total_params}M")
        # Save the configuration of flow_model to a json file
        config_dict = flow_model.config
        config_dict['num_params'] = pytorch_total_params
        with open(os.path.join(arg.dir, 'config_flow_model.json'), 'w') as f:
            json.dump(config_dict, f, indent = 4)
    if arg.resume is not None:
        state_dict = torch.load(arg.resume, map_location = 'cpu')
    else:
        state_dict = None


    ################################## FLOW MODEL AND FORWARD MODEL #########################################

    flow_model = flow_model.to(device)
    # flow_model = torch.compile(flow_model)
    flow_model = DDP(flow_model, device_ids=[rank])
    if not arg.no_ema:
        ema_flow_model = EMAMODEL(model=flow_model)
    else:
        ema_flow_model = None

    ################################### Learning Optimizer ###################################################
    learnable_params = []
    learnable_params += list(flow_model.parameters())
    if arg.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(learnable_params, lr=arg.learning_rate, weight_decay=0.1, betas = (0.9, 0.9999))
    elif arg.optimizer == 'adam':
        optimizer = torch.optim.Adam(learnable_params, lr=arg.learning_rate, betas = (0.9, 0.999), eps=1e-8)
    else:
        raise NotImplementedError
    if rank==0:
        print(f"Start training")
        
    distill(rank,flow_model, data_loader, arg.iterations, optimizer, data_shape,device,arg,state_dict=state_dict,ema_flow_model=ema_flow_model)
    destroy_process_group()

if __name__ == "__main__":
    arg = get_args()
    if not os.path.exists(arg.dir):
        os.makedirs(arg.dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu
    device_ids = arg.gpu.split(',')
    device_ids = [int(i) for i in device_ids]
    world_size = len(device_ids)
    with open(os.path.join(arg.dir, "config.json"), "w") as json_file:
        json.dump(vars(arg), json_file, indent = 4)
    arg.batchsize = arg.batchsize // world_size
    try:
       mp.spawn(main, args=(world_size, arg), nprocs=world_size)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        destroy_process_group()
        exit(0)
