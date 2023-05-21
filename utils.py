import torch
import numpy as np
from torch.distributions import Normal, Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os


def get_train_data(COMP, D, VAR):
    initial_mix = Categorical(torch.tensor([1/COMP for i in range(COMP)]))
    initial_comp = MultivariateNormal(torch.tensor([[D * np.sqrt(3) / 2., D / 2.], [-D * np.sqrt(3) / 2., D / 2.], [0.0, - D * np.sqrt(3) / 2.]]).float(), VAR * torch.stack([torch.eye(2) for i in range(COMP)]))
    initial_model = MixtureSameFamily(initial_mix, initial_comp)
    samples_1 = initial_model.sample([1000000])

    target_mix = Categorical(torch.tensor([1/COMP for i in range(COMP)]))
    target_comp = MultivariateNormal(torch.tensor([[D * np.sqrt(3) / 2., - D / 2.], [-D * np.sqrt(3) / 2., - D / 2.], [0.0, D * np.sqrt(3) / 2.]]).float(), VAR * torch.stack([torch.eye(2) for i in range(COMP)]))
    target_model = MixtureSameFamily(target_mix, target_comp)
    samples_2 = target_model.sample([100000])

    return samples_1, samples_2

def get_train_data_two_gaussian(mu1, mu2, cov1, cov2):
    # mu_val = 8.
    # cov_val = 0.
    # mu1 = torch.tensor([-mu_val, mu_val])
    # mu2 = torch.tensor([mu_val, mu_val])
    # cov1 = torch.tensor([[3, cov_val], [cov_val, .1]])
    # cov2 = torch.tensor([[3, -cov_val], [-cov_val, .1]])

    dist1 = MultivariateNormal(mu1, cov1)
    dist2 = MultivariateNormal(mu2, cov2)

    samples_1 = dist1.sample([100000])
    samples_2 = dist2.sample([100000])

    return samples_1, samples_2


@torch.no_grad()
def draw_plot(x1, x2, x_recon1, x_recon2, z1, z2, i, DOT_SIZE, M, dir, mu_prior, var_prior):
    z = np.random.normal(size=z1.shape)
    z = mu_prior + var_prior ** 0.5 * z
    x1 = x1.detach().cpu().numpy()
    x2 = x2.detach().cpu().numpy()
    x_recon1 = x_recon1.detach().cpu().numpy()
    x_recon2 = x_recon2.detach().cpu().numpy()
    z1 = z1.detach().cpu().numpy()
    z2 = z2.detach().cpu().numpy()

    # Draw x1, x2, x_recon1, x_recon2 with labels
    plt.figure(figsize=(4, 4))
    plt.scatter(x1[:, 0], x1[:, 1], alpha=0.15, color="red", s = DOT_SIZE*3)
    plt.scatter(x2[:, 0], x2[:, 1], alpha=0.15, color="orange", s = DOT_SIZE*3)
    plt.scatter(x_recon1[:, 0], x_recon1[:, 1], alpha=0.15, color="blue", s = DOT_SIZE*3)
    plt.scatter(x_recon2[:, 0], x_recon2[:, 1], alpha=0.15, color="green", s = DOT_SIZE*3)
    plt.xlim(-M, M)
    plt.ylim(-M, M)
    plt.legend(["x1", "x2", "x_recon1", "x_recon2"])
    plt.title("x1, x2, x_recon1, x_recon2")
    plt.savefig(os.path.join(dir, f"recon_{i}.jpg"))

    # Draw z, z1, z2 with labels
    plt.figure(figsize=(4, 4))
    plt.scatter(z[:, 0], z[:, 1], alpha=0.15, color="black", s = DOT_SIZE)
    plt.scatter(z1[:, 0], z1[:, 1], alpha=0.15, color="red", s = DOT_SIZE)
    plt.scatter(z2[:, 0], z2[:, 1], alpha=0.15, color="orange", s = DOT_SIZE)
    plt.xlim(-M, M)
    plt.ylim(-M, M)
    plt.legend(["z", "z1", "z2"])
    plt.title("z, z1, z2")
    plt.savefig(os.path.join(dir, f"z_{i}.jpg"))

    # close all figures
    plt.close('all')
def cosine_similarity(x1, x2):
    x1 = x1.view(x1.shape[0], -1)
    x2 = x2.view(x2.shape[0], -1)
    x1_norm = x1 / x1.norm(dim=1, keepdim=True)
    x2_norm = x2 / x2.norm(dim=1, keepdim=True)
    return torch.sum(x1_norm * x2_norm, dim=1).mean()

def straightness(traj):
    N = len(traj) - 1
    dt = 1 / N
    base = traj[0] - traj[-1]
    mse = []
    for i in range(1, len(traj)):
        v = (traj[i-1] - traj[i]) / dt
        mse.append(torch.mean((v - base) ** 2))
    return torch.mean(torch.stack(mse))

def straightness_no_mean(traj):
    N = len(traj) - 1
    dt = 1 / N
    base = traj[0] - traj[-1]
    mse = []
    for i in range(1, len(traj)):
        v = (traj[i-1] - traj[i]) / dt
        mse.append(cosine_similarity(v,base))
    return torch.stack(mse)


def get_kl(mu, logvar):
    # Return KL divergence between N(mu, var) and N(0, 1), divided by data dimension.
    kl = 0.5 * torch.sum(-1 - logvar + mu.pow(2) + logvar.exp(), dim=[1,2,3])
    loss_prior = torch.mean(kl) / (mu.shape[1]*mu.shape[2]*mu.shape[3])
    return loss_prior
def get_kl_2d(mu, logvar, wide_prior = True):
    if wide_prior:
        kl = 0.5 * torch.sum(-1 + np.log(36) - logvar + mu.pow(2) / torch.tensor([36,1], device=mu.device) + logvar.exp() / torch.tensor([36,1], device=mu.device), dim=1)
    else:
        # Return KL divergence between N(mu, var) and N(0, 1), divided by data dimension.
        kl = 0.5 * torch.sum(-1 - logvar + mu.pow(2) + logvar.exp(), dim=[1])
    loss_prior = torch.mean(kl) / 2
    return loss_prior
def get_kl_2d_gen(mu1, logvar1, mu2, var2):
    # Generalized KL divergence between N(mu1, var1) and N(mu2, var2), divided by data dimension.
    mu2, var2 = mu2.unsqueeze(0), var2.unsqueeze(0)
    # Return KL divergence between N(mu1, var1) and N(mu2, var2), divided by data dimension.
    kl = 0.5 * (torch.sum(torch.log(var2), dim = 1) - torch.sum(logvar1, dim = 1) - 2 + torch.sum((mu1 - mu2) ** 2 / var2, dim = 1) + torch.sum(logvar1.exp() / var2, dim = 1))
    # kl = 0.5 * torch.sum(-1 + np.log(36) - logvar1 + mu1.pow(2) / torch.tensor([36,1], device=mu1.device) + logvar1.exp() / torch.tensor([36,1], device=mu1.device), dim=1)

    loss_prior = torch.mean(kl) / 2
    return loss_prior

def alpha(t):
    # DDPM defines x_t(x, z) = alpha(t)x + sqrt(1 - alpha(t)^2)z
    a = 19.9
    b = 0.1
    exp = torch.exp if isinstance(t, torch.Tensor) else np.exp
    return exp(-0.25 * a * t ** 2 - 0.5 * b * t)
def dalpha_dt(t):
    a = 19.9
    b = 0.1
    alpha_t = alpha(t)
    return (-0.5 * a * t - 0.5 * b) * alpha_t
def d_1_minus_alpha_sq_dt(t):
    a = 19.9
    b = 0.1
    alpha_t = alpha(t)
    return 0.5 * (1 - alpha_t ** 2) ** (-0.5) * (-2 * alpha_t) * dalpha_dt(t)
def convert_ddp_state_dict_to_single(ddp_state_dict):
    single_state_dict = {}
    for key, value in ddp_state_dict.items():
        # 去掉 "module." 前缀
        new_key = key.replace('module.', '') if key.startswith('module.') else key
        new_key = new_key.replace('_orig_mod.', '') if new_key.startswith('_orig_mod.') else new_key
        single_state_dict[new_key] = value
    return single_state_dict
def RK(function,value,start,step,step_nums,method="RK45"):
    if method == "RK67":
        def rk67_fixed_step(ode, y0, t0, dt, num_steps):
            y = y0
            t = t0
            for _ in range(num_steps):
                k1 = ode(t, y)
                k2 = ode(t + dt / 5, y + dt * k1 / 5)
                k3 = ode(t + 3 * dt / 10, y + 3 * dt * k1 / 40 + 9 * dt * k2 / 40)
                k4 = ode(t + 4 * dt / 5, y + 44 * dt * k1 / 45 - 56 * dt * k2 / 15 + 32 * dt * k3 / 9)
                k5 = ode(t + 8 * dt / 9, y + 19372 * dt * k1 / 6561 - 25360 * dt * k2 / 2187 + 64448 * dt * k3 / 6561 - 212 * dt * k4 / 729)
                k6 = ode(t + dt, y + 9017 * dt * k1 / 3168 - 355 * dt * k2 / 33 + 46732 * dt * k3 / 5247 + 49 * dt * k4 / 176 - 5103 * dt * k5 / 18656)
                y += dt * (35 * k1 / 384 + 500 * k3 / 1113 + 125 * k4 / 192 - 2187 * k5 / 6784 + 11 * k6 / 84)
                t += dt
            return y
        return rk67_fixed_step(function,value,start,step,step_nums)
    elif method == "RK45":
        def rk45_fixed_step(ode, y0, t0, dt, num_steps):
            y = y0
            t = t0
            for _ in range(num_steps):
                k1 = ode(t, y)
                k2 = ode(t + dt/2, y + dt * k1/2)
                k3 = ode(t + dt/2, y + dt * k2/2)
                k4 = ode(t + dt, y + dt * k3)
                y += dt * (k1 + 2*k2 + 2*k3 + k4) / 6
                t += dt
            return y
        return rk45_fixed_step(function,value,start,step,step_nums)
    elif method == "RK23":
        def rk23_fixed_step(ode, y0, t0, dt, num_steps):
            y = y0
            t = t0
            for _ in range(num_steps):
                k1 = ode(t, y)
                k2 = ode(t + dt, y + dt * k1)
                y += dt * (k1 + k2) / 2
                t += dt
            return y
        return rk23_fixed_step(function,value,start,step,step_nums)
    else:
        raise NotImplementedError

class InfiniteDataLoaderIterator:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            batch = next(self.data_iter)
        return batch
    
import torch
import torch.nn as nn
import torchvision.models as models

class LPIPS(nn.Module):
    def __init__(self):
        super(LPIPS, self).__init__()
        self.net = models.vgg16(pretrained=True).features[:29]
        for param in self.net.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x = self.net(x)
        y = self.net(y)
        return torch.mean(torch.square(x - y))
    
from PIL import Image
from numpy import random
def merge_images(folder_path, output_path, n, m):
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
    indices = list(range(len(image_files))) # indices = the number of images in the source data set
    random.shuffle(indices)
    image_files = [image_files[i] for i in indices]
    image_files = image_files[:n * m]

    if len(image_files) < n * m:
        print("Warning: Not enough images found. Using available images.")

    images = [Image.open(os.path.join(folder_path, img_file)) for img_file in image_files]

    # 获取最大宽度和高度
    max_width = max([img.width for img in images])
    max_height = max([img.height for img in images])

    # 创建一个空白图片作为拼接结果
    result = Image.new('RGB', ((max_width+2) * n, (max_height+2) * m))

    # 按行列拼接图片
    for i in range(m):
        for j in range(n):
            index = i * n + j
            if index < len(images):
                img = images[index]
                result.paste(img, (j * (max_width+2), i * (max_height+2)))

    # 保存拼接后的图片
    result.save(output_path)
    
if __name__=="__main__":
    for i in range(7):
        merge_images(f"./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_8/trajs_{i}/", f"./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_8/sample_trajs_{i}.png",15,15)