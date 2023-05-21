# From https://colab.research.google.com/drive/1LouqFBIC7pnubCOl5fhnFd33-oVJao2J?usp=sharing#scrollTo=yn1KM6WQ_7Em

import torch
import numpy as np
from flows import ResidualFlow,OnlineSlimFlow
import torch.nn as nn
import tensorboardX
import os
from models import UNetEncoder
from guided_diffusion.unet import UNetModel
import torchvision.datasets as dsets
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from utils import straightness
from dataset import CelebAHQImgDataset
import argparse
from tqdm import tqdm
from network_edm import SongUNet
from torch.nn import DataParallel
import json,copy
from utils import convert_ddp_state_dict_to_single,InfiniteDataLoaderIterator
from train_cud_reverse_img_ddp import parse_config
"""
python eval.py --gpu 0 --dir ./runs/cifar10-onlineslim-predstep-1-noema-beta20/ --N 16 --res 32 \
      --input_nc 3 --num_samples 500 --ckpt ./runs/cifar10-onlineslim-predstep-1-noema-beta20/flow_model_500000_ema.pth \
      --config_en configs/cifar10_en.json --config_de configs/cifar10_de.json --dataset cifar10
"""
import einops
def get_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Configs')
    parser.add_argument('--gpu', type=str, help='gpu index')
    parser.add_argument('--dir', type=str, help='Saving directory name')
    parser.add_argument('--ckpt', type=str, default = None, help='Flow network checkpoint')
    parser.add_argument('--batchsize', type=int, default = 64, help='Batch size')
    parser.add_argument('--res', type=int, default = 64, help='Image resolution')
    parser.add_argument('--input_nc', type=int, default = 3, help='Unet num_channels')
    parser.add_argument('--eval_mode', type=str, default = "id", help='if sample')
    parser.add_argument('--N', type=int, default = 4, help='Number of sampling steps')
    parser.add_argument('--num_samples', type=int, default = 10000, help='Number of samples to generate')
    parser.add_argument('--begin_sample_index', type=int, default = 0, help='starting position of sampling')
    parser.add_argument('--encoder', type=str, default=None,  help='If Use Encoder')
    parser.add_argument('--dataset', type=str, help='cifar10 / mnist / celebahq')
    parser.add_argument('--no_scale', action='store_true', help='Store true if the model is trained on [0,1] scale')    
    parser.add_argument('--save_traj', action='store_true', help='Save the trajectories')    
    parser.add_argument('--save_z', action='store_true', help='Save zs for distillation')    
    parser.add_argument('--save_data', action='store_true', help='Save data')    
    parser.add_argument('--solver', type=str, default = 'euler', help='ODE solvers')
    parser.add_argument('--config_de', type=str, default = None, help='Decoder config path, must be .json file')
    parser.add_argument('--config_en', type=str, default = None, help='Encoder config path, must be .json file')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--rtol', type=float, default=1e-5, help='rtol for RK45 solver')
    parser.add_argument('--atol', type=float, default=1e-5, help='atol for RK45 solver')
    


    arg = parser.parse_args()
    return arg

def main(arg):
    if not os.path.exists(arg.dir):
        os.makedirs(arg.dir)
    assert arg.config_de is not None
    config = parse_config(arg.config_de)
    if not os.path.exists(os.path.join(arg.dir, f"eval")):
        os.makedirs(os.path.join(arg.dir, f"eval"))
    

    if config['unet_type'] == 'adm':
        model_class = UNetModel
    elif config['unet_type'] == 'songunet':
        model_class = SongUNet
    # Pass the arguments in the config file to the model
    flow_model = model_class(**config)
    device_ids = arg.gpu.split(',')
    model_list = []
    if arg.ckpt is not None:
        flow_training_state = convert_ddp_state_dict_to_single(torch.load(arg.ckpt,map_location="cpu"))# ["model_state_dict"][0]
        flow_model.load_state_dict(flow_training_state)
    if len(device_ids) > 1:
        device = torch.device(f"cuda")
        print(f"Using {device_ids} GPUs!")
        flow_model = DataParallel(flow_model)
    else:
        device = torch.device(f"cuda")
        print(f"Using GPU {arg.gpu}!")
    # Print the number of parameters in the model
    flow_model = flow_model.to(device)
    rectified_flow = OnlineSlimFlow(device, flow_model,None, num_steps = 100)
    flow_model.eval()

    if arg.encoder is not None:
        config_en = parse_config(arg.config_en)
        if config_en['unet_type'] == 'adm':
            encoder_class = UNetModel
        elif config_en['unet_type'] == 'songunet':
            encoder_class = SongUNet
        # Pass the arguments in the config file to the model
        encoder = encoder_class(**config_en)
        # encoder = SongUNet(img_resolution = arg.res, in_channels = arg.input_nc, out_channels = arg.input_nc * 2, channel_mult = [2,2,2], dropout = 0.13, num_blocks = 2, model_channels = 32)
        forward_model = UNetEncoder(encoder = encoder, input_nc = arg.input_nc)
        forward_model = torch.compile(forward_model,backend="inductor")
        forward_model.load_state_dict(convert_ddp_state_dict_to_single(torch.load(arg.encoder,map_location='cpu')), strict = True)
        forward_model = forward_model.to(device).eval()
    from train_cud_reverse_img_ddp import get_loader
    data_loader, _, _, _ = get_loader(arg.dataset, arg.batchsize, 1, 0)
    train_iter = InfiniteDataLoaderIterator(data_loader)
    # Save configs as json file
    config_dict = vars(arg)
    with open(os.path.join(arg.dir, 'config_sampling.json'), 'w') as f:
        json.dump(config_dict, f, indent = 4)
    with torch.no_grad():
        i = 0
        epoch = arg.num_samples // arg.batchsize + 1
        save_list = [[] for i in range(100)]
        for ep in tqdm(range(epoch)):
            noise = torch.randn(arg.batchsize, arg.input_nc, arg.res, arg.res).to(device)
            x, _= next(train_iter)
            x = x.to(device)
            if arg.encoder is not None:
                noise = noise[1].repeat(arg.batchsize, 1, 1, 1)
                z, _, _ = forward_model(x, noise = noise)
            else:
                z = noise
            for lt in range(1,101):
                t = torch.ones(x.shape[0],).to(x.device).float() * (lt)/100
                dis = rectified_flow.get_eval_dis(x,z,t,100).abs().mean(dim=[1,2,3]).tolist()
                save_list[lt-1].extend(dis)
        for i in range(len(save_list)):
            save_list[i] = sum(save_list[i])/len(save_list[i])
        fig_x = np.arange(0,len(save_list))
        fig_y = save_list
        print(fig_y)
        import matplotlib.pyplot as plt
        plt.plot(fig_x,fig_y,"--",linewidth=3)
        plt.savefig(os.path.join(arg.dir, "eval", f"dis.png"))
    
        

if __name__ == "__main__":
    arg = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu
    torch.manual_seed(arg.seed)
    print(f"seed: {arg.seed}")
    main(arg)