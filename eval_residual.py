# From https://colab.research.google.com/drive/1LouqFBIC7pnubCOl5fhnFd33-oVJao2J?usp=sharing#scrollTo=yn1KM6WQ_7Em

import torch
import numpy as np
from flows import ResidualFlow
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
from train_reverse_img_ddp import parse_config
import einops
def get_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Configs')
    parser.add_argument('--gpu', type=str, help='gpu index')
    parser.add_argument('--dir', type=str, help='Saving directory name')
    parser.add_argument('--ckpt', type=str, default = None, help='Flow network checkpoint')
    parser.add_argument('--batchsize', type=int, default = 4, help='Batch size')
    parser.add_argument('--res', type=int, default = 64, help='Image resolution')
    parser.add_argument('--input_nc', type=int, default = 3, help='Unet num_channels')
    parser.add_argument('--eval_mode', type=str, default = "id", help='if sample')
    parser.add_argument('--N', type=int, default = 4, help='Number of sampling steps')
    parser.add_argument('--num_samples', type=int, default = 64, help='Number of samples to generate')
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
    flow_model = torch.compile(flow_model,backend="inductor")
    device_ids = arg.gpu.split(',')
    model_list = []
    if arg.ckpt is not None:
        flow_training_state = torch.load(arg.ckpt,map_location="cpu")
        for ii in range(len(flow_training_state)):
            sub_flow_model = copy.deepcopy(flow_model)
            sub_flow_model.load_state_dict(convert_ddp_state_dict_to_single(flow_training_state[ii]))
            model_list.append(sub_flow_model)
    else:
        raise NotImplementedError("Model ckpt should be provided.")
    if len(device_ids) > 1:
        device = torch.device(f"cuda")
        print(f"Using {device_ids} GPUs!")
        for ii in range(len(model_list)):
            model_list[ii] = DataParallel(model_list[ii])
    else:
        device = torch.device(f"cuda")
        print(f"Using GPU {arg.gpu}!")
    # Print the number of parameters in the model
    pytorch_total_params = sum(sum(p.numel() for p in sub_flow_model.parameters()) for sub_flow_model in model_list)
    # Convert to M
    pytorch_total_params = pytorch_total_params / 1000000
    print(f"Total number of parameters: {pytorch_total_params}M")
    for ii in range(len(model_list)):
        model_list[ii] = model_list[ii].to(device)
    rectified_flow = ResidualFlow(device, model_list, num_steps = arg.N)
    for sub_flow_model in rectified_flow.model_list:
        sub_flow_model.eval()

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
    from train_residual_reverse_img_ddp import get_loader
    data_loader, _, _, _ = get_loader(arg.dataset, arg.batchsize, 1, 0)
    train_iter = InfiniteDataLoaderIterator(data_loader)
    # Save configs as json file
    config_dict = vars(arg)
    with open(os.path.join(arg.dir, 'config_sampling.json'), 'w') as f:
        json.dump(config_dict, f, indent = 4)
    with torch.no_grad():
        if arg.eval_mode == "id":
            i = 0
            epoch = arg.num_samples // arg.batchsize + 1
            residual_number = 2
            save_list = [[] for i in range(residual_number*17)]
            for ep in tqdm(range(epoch)):
                noise = torch.randn(arg.batchsize, arg.input_nc, arg.res, arg.res).to(device)
                x, _= next(train_iter)
                x = x.to(device)
                if arg.encoder is not None:
                    # x = x[1].repeat(arg.batchsize, 1, 1, 1)
                    noise = noise[1].repeat(arg.batchsize, 1, 1, 1)
                    z, _, _ = forward_model(x, noise = noise)
                else:
                    z = noise
                for res in range(residual_number):
                    for lt in range(17):
                        t = torch.ones(x.shape[0],).to(x.device).float() * (lt)/16
                        dis = rectified_flow.get_eval_dis(x,z,t,res,2).abs().mean(dim=[1,2,3]).tolist()
                        save_list[res*17+lt].extend(dis)
            for i in range(len(save_list)):
                save_list[i] = sum(save_list[i])/len(save_list[i])
            fig_x = np.arange(0,len(save_list))
            fig_y = save_list
            import matplotlib.pyplot as plt
            plt.plot(fig_x,fig_y,"--x")
            plt.savefig(os.path.join(arg.dir, "eval", f"dis.png"))
        else:
            i = 0
            begin_sample_index = arg.begin_sample_index
            if begin_sample_index <0:
                begin_sample_index = 0
            epoch = arg.num_samples // arg.batchsize + 1
            residual_number = 2
            save_list = [[] for i in range(residual_number*17)]

            for ep in tqdm(range(epoch)):
                noise = torch.randn(arg.batchsize, arg.input_nc, arg.res, arg.res).to(device)
                x, _= next(train_iter)
                x = x.to(device)
                if arg.encoder is not None:
                    # x = x[1].repeat(arg.batchsize, 1, 1, 1)
                    noise = noise[1].repeat(arg.batchsize, 1, 1, 1)
                    z, _, _ = forward_model(x, noise = noise)
                else:
                    z = noise
                begin_noise = z.clone().detach()
                new_index = residual_number*17 - begin_sample_index - 1
                bt = new_index%17
                t = torch.ones(x.shape[0],).to(x.device).float() * bt / 16
                t = t.view(t.shape[0],1,1,1)
                end = int(begin_sample_index//17)
                z = rectified_flow.sample_residual_ode(z,16,end=end)
                z = t * z + (1.-t) * x
                dt = 1 / arg.N
                traj = [z]
                gt_traj = [z]
                for index in range(begin_sample_index,residual_number*17):
                    new_index = residual_number*17 - index - 1
                    if new_index%17==0:
                        continue
                    res = residual_number - int(new_index//17) - 1
                    lt = new_index%17
                    t = torch.ones(x.shape[0],).to(x.device).float() * lt / 16
                    vt = rectified_flow.model_list[res](z, t.squeeze())
                    z = z.detach().clone() - vt * dt
                    pre_index = residual_number*17 - index - 2
                    if pre_index < 0:
                        gt_z = z
                    else:
                        res = residual_number - int(pre_index//17) - 1
                        lt = pre_index%17
                        t = torch.ones(x.shape[0],).to(x.device).float() * lt / 16
                        t = t.view(t.shape[0],1,1,1)
                        end = int((index+2)//17)
                        tmp_z = rectified_flow.sample_residual_ode(begin_noise,16,end=end)
                        gt_z = t * tmp_z + (1.-t) * x
                    gt_z = gt_z.clone().detach()
                    dis_l = (z - gt_z).abs().mean(dim=[1,2,3]).view(-1).tolist()
                    gt_traj.append(gt_z.clone())
                    traj.append(z.clone())
                    save_list[index].extend(dis_l)
                traj = einops.rearrange(torch.stack(traj,dim=0)[:,:4,...],"a b c d e -> c (a d) (b e)").contiguous()
                save_image(traj*0.5+0.5,os.path.join(arg.dir, f"eval","traj.png"))
                # N,B,3,H,W
                gt_traj = einops.rearrange(torch.stack(gt_traj,dim=0)[:,:4,...],"a b c d e -> c (a d) (b e)").contiguous()
                save_image(gt_traj*0.5+0.5,os.path.join(arg.dir, f"eval","gt_traj.png"))
                exit(-1)
            for i in range(len(save_list)):
                if len(save_list[i])!=0:
                    save_list[i] = sum(save_list[i])/len(save_list[i])
                else:
                    save_list[i] = 0
            fig_x = np.arange(0,len(save_list))
            fig_y = save_list
            import matplotlib.pyplot as plt
            plt.plot(fig_x,fig_y,"--x")
            plt.savefig(os.path.join(arg.dir, "eval", f"dis_2.png"))
        
        

if __name__ == "__main__":
    arg = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu
    torch.manual_seed(arg.seed)
    print(f"seed: {arg.seed}")
    main(arg)