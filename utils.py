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

# if __name__=="__main__":
#     A = [0.6913336895522662, 0.632988414959982, 0.5818285319837742, 0.5313371390220709, 0.49181826095446013, 0.46558589951018803, 0.44220674701500684, 0.4202135790983448, 0.40102913284499664, 0.3845856217376422, 0.3703993319941219, 0.35802926021278836, 0.3471334888745332, 0.33747194983880036, 0.3288580815860769, 0.32114029524382204, 0.3141936414176598, 0.30791368712380063, 0.30221791214717086, 0.2970356437508599, 0.29231149969564285, 0.2880010176595533, 0.28406113357050344, 0.2804498991317814, 0.2771383810977568, 0.2740976731001865, 0.27130677634704625, 0.2687434173640213, 0.26639323213748867, 0.2642379854223691, 0.2622609430400189, 0.2604532346013002, 0.25880719787528506, 0.2573159161474905, 0.25596966292505385, 0.2547628957181587, 0.25369477498315973, 0.2527426669621491, 0.25191089079453377, 0.2511975105153397, 0.2505959521076875, 0.2501040941424435, 0.24972515285480767, 0.24945602837397018, 0.24928343411738751, 0.2492053553214646, 0.24922486806462985, 0.2493471921843593, 0.24956416174245533, 0.24987148070067633, 0.25027724181563826, 0.25076929614442633, 0.2513505338210962, 0.2520359552145237, 0.2528306060048635, 0.25376151632372057, 0.25480424299894366, 0.25590962947899243, 0.2571223629201995, 0.25845076840778347, 0.2598729122255463, 0.261393870372558, 0.2630013393063564, 0.26479753010789864, 0.26675646518560825, 0.26875921726605156, 0.2708032991795335, 0.2729573672550032, 0.2752659727266291, 0.2776831385999685, 0.28025367102236487, 0.2829046992555959, 0.2855579815659439, 0.2883118861063849, 0.2912477659410797, 0.294350747964927, 0.2976432060531806, 0.3011397400550777, 0.3048480076540727, 0.30874270701315254, 0.31272131791047286, 0.3168490096286405, 0.32120731608301867, 0.32582105848996434, 0.33069293209700845, 0.335938605494448, 0.3415454721689457, 0.34744170485646464, 0.3536487888341071, 0.3602102364238817, 0.3671600816596765, 0.3745755797572201, 0.38248900409962516, 0.3909487891360186, 0.3999888204707531, 0.4095695087016793, 0.41972831975726876, 0.4302381530869752, 0.4407613236398902, 0.4512418095546309]
#     B = [0.5681934047315735, 0.5079279888741439, 0.4694957007159246, 0.4395563930593198, 0.41555789264384657, 0.3959448299428914, 0.3796262281393865, 0.3658063290058635, 0.35395514427364105, 0.3436717245131149, 0.3346593253372703, 0.3268166410853155, 0.32471856936172117, 0.31734015711117536, 0.3107941285925335, 0.30509398727008374, 0.3000340487851645, 0.29608871583332075, 0.29498223828704795, 0.29107013038446894, 0.28719470734358765, 0.28383981537626823, 0.28084178022982087, 0.27814794042205904, 0.27648997298820177, 0.27534736955567496, 0.2727676439535571, 0.270414445141796, 0.2685533988333191, 0.26697524181508925, 0.2655382275916054, 0.2642103064499679, 0.2629845992923947, 0.26185304464161163, 0.2608169833620195, 0.25987238703964977, 0.2590150073665427, 0.25824251432641177, 0.2575601544085657, 0.25697304939239984, 0.256466703111073, 0.2560467567964224, 0.255736950384744, 0.2555217128392542, 0.255370151338866, 0.25528299819416134, 0.2552744925560546, 0.25536657331394963, 0.2555558992244187, 0.2558343502896605, 0.2562057787072263, 0.2566731971601257, 0.25720998540055007, 0.2578219592105597, 0.2585018793688505, 0.2593011664648657, 0.26020370092010126, 0.26118862167641055, 0.26225205477385316, 0.2633888145937817, 0.26469443380483426, 0.2661345667002024, 0.26761198152962606, 0.2691750907106325, 0.2708871672657551, 0.2726876057131449, 0.2746107851417037, 0.27661038588848896, 0.27870144856569823, 0.2809044842579169, 0.2832250978972297, 0.28568190764053725, 0.28828009140852373, 0.2909486338903662, 0.293750110140536, 0.2967717490682844, 0.3000167901191162, 0.30344172328477725, 0.3070493178529432, 0.31080740576726384, 0.3147485259833047, 0.31884833448566496, 0.32324018664075993, 0.3279799616429955, 0.33302321373776067, 0.3383648143062601, 0.3440481340221595, 0.34999382257228717, 0.3562683797063073, 0.36292116866388824, 0.369914147304371, 0.3773563539143652, 0.385327302486985, 0.39389420753286686, 0.4030871444119839, 0.41274651075946167, 0.4228822146978928, 0.43352404062170535, 0.4444257758295862, 0.4560618831892498]
#     C = [0.564631033805199, 0.5058440497232368, 0.4682503718213411, 0.4386642431927612, 0.4146258576074615, 0.3948606150224805, 0.37830027713789605, 0.36421250106650405, 0.35207828987040557, 0.34150723305356223, 0.332194106602401, 0.32391299928713124, 0.31649482275679475, 0.3098146330739837, 0.3037675181258237, 0.2982688221454737, 0.29324585169524653, 0.28863467049086466, 0.28439074318885105, 0.2804789061556221, 0.27686827481375076, 0.2735309659547056, 0.2704491672629956, 0.2675948963587871, 0.2649454236452584, 0.26248409725667443, 0.2602012507340987, 0.258088016671536, 0.2561364950379357, 0.2543412775703473, 0.2526924730700557, 0.25117943048098823, 0.24979274787619943, 0.2485270654578926, 0.24738080997485667, 0.24634537785459543, 0.2454184926027665, 0.24459949009178672, 0.24388962564989924, 0.24328637081634952, 0.24278750593657605, 0.2423901575821219, 0.24208427565463353, 0.24186826169898268, 0.2417454563328647, 0.2417222031581332, 0.24180436821916373, 0.24199602941371268, 0.24229137780639576, 0.2426827638046234, 0.24316141815506853, 0.24373707250197185, 0.2444255407244782, 0.24522160528431414, 0.24613089695048984, 0.24715999047475634, 0.24828729939326877, 0.24949462015501922, 0.25077994738239795, 0.25217026748578064, 0.25365421480091754, 0.25523588994838065, 0.2569767499808222, 0.2587799720276962, 0.26066110467218095, 0.2626493978095823, 0.2647879222058691, 0.26714644227467943, 0.26966134170652367, 0.27225301222642884, 0.2749720768770203, 0.27778131172817666, 0.2806623968135682, 0.28367588082619477, 0.28680962207727134, 0.2900799399212701, 0.29354592506570043, 0.2972187396517256, 0.3010065524795209, 0.3049185904528713, 0.30900403635314433, 0.31332675978774205, 0.31791405951662455, 0.32271056638273876, 0.3277784041711129, 0.3330230402207235, 0.3384384550881805, 0.3440931480145082, 0.34996388528088573, 0.3560869343491504, 0.36252577416598797, 0.3692871130333515, 0.3763530666328734, 0.383699421377969, 0.39126661070622504, 0.3989044174377341, 0.40633321333734784, 0.41316009996808134, 0.4190465370338643, 0.42393766266468447]
#     x = np.linspace(0.01,1,100)
#     import matplotlib.pyplot as plt
#     plt.plot(x, A,"--",linewidth=3,label="predstep=1")
#     plt.plot(x, B,"--",linewidth=3,label="predstep=2")
#     plt.plot(x, C,"--",linewidth=3,label="predstep=0")
#     plt.legend()
#     plt.savefig("./test.png")

if __name__=="__main__":
    for i in range(7):
        merge_images(f"./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_8/trajs_{i}/", f"./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_8/sample_trajs_{i}.png",15,15)