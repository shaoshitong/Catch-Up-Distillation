import torch
from tqdm import tqdm
from utils import alpha, dalpha_dt, d_1_minus_alpha_sq_dt
from scipy import integrate
from dpm_solver_pytorch import DPM_Solver, model_wrapper, NoiseScheduleVP
from utils import RK
import functools

class BaseFlow():
  def __init__(self, device, model=None, ema_model=None, num_steps=1000):
    self.model = model
    self.ema_model = ema_model
    self.N = num_steps
    self.device = device
  
  def get_train_tuple(self, z0=None, z1=None):
    # return z_t, t, target
    raise NotImplementedError(f"get_train_tuple not implemented for {self.__class__.__name__}.")
    
  @torch.no_grad()
  def sample_ode(self, z0=None, N=None):
    ### NOTE: Use Euler method to sample from the learned flow
    if N is None:
      N = self.N    
    dt = 1./N
    traj = [] # to store the trajectory
    z = z0.detach().clone()
    batchsize = z.shape[0]
    
    traj.append(z.detach().clone())
    for i in range(N):
      t = torch.ones((batchsize,1), device=self.device) * i / N
      if len(z0.shape) == 2:
        pred = self.model(z, t)
      elif len(z0.shape) == 4:
        pred = self.model(z, t.squeeze())
      z = z.detach().clone() + pred * dt
      
      traj.append(z.detach().clone())
    return traj

  @torch.no_grad()
  def sample_ode_generative(self, z1=None, N=None, use_tqdm=True, solver = 'euler'):
    assert solver in ['euler', 'heun']
    tq = tqdm if use_tqdm else lambda x: x
    if N is None:
      N = self.N    
    if solver == 'heun' and N % 2 == 0:
      raise ValueError("N must be odd when using Heun's method.")
    if solver == 'heun':
      N = (N + 1) // 2
    dt = -1./N
    traj = [] # to store the trajectory
    x0hat_list = []
    z = z1.detach().clone()
    batchsize = z.shape[0]
    
    traj.append(z.detach().clone())
    for i in tq(reversed(range(1,N+1))):
      t = torch.ones((batchsize,1), device=self.device) * i / N
      t_next = torch.ones((batchsize,1), device=self.device) * (i-1) / N
      if len(z1.shape) == 2:
        if solver == 'heun':
          raise NotImplementedError("Heun's method not implemented for 2D data.")
        vt = self.model(z, t)
      elif len(z1.shape) == 4:
        vt = self.model(z, t.squeeze())
        if solver == 'heun' and i > 1:
          z_next = z.detach().clone() + vt * dt
          vt_next = self.model(z_next, t_next.squeeze())
          vt = (vt + vt_next) / 2
        x0hat = z - vt * t.view(-1,1,1,1)
        x0hat_list.append(x0hat)
      
        
      z = z.detach().clone() + vt * dt
      
      traj.append(z.detach().clone())
    return traj, x0hat_list
  
  def sample_ode_generative_bbox(self, z1=None, N=None, use_tqdm=True, solver = 'RK45', eps = 1e-3, rtol=1e-5, atol=1e-5,):
    dshape = z1.shape
    device = z1.device
    def ode_func(t, x):
      x = torch.from_numpy(x.reshape(dshape)).to(device).type(torch.float32)
      vec_t = torch.ones(dshape[0], device=x.device) * t
      vt = self.model(x, vec_t)
      vt = vt.detach().cpu().numpy().reshape(-1)
      return vt
    solution = integrate.solve_ivp(ode_func, (1, eps), z1.detach().cpu().numpy().reshape(-1), method=solver, rtol = rtol, atol = atol)
    nfe = solution.nfev
    result = torch.from_numpy(solution.y[:,-1].reshape(dshape))
    return result, nfe

  def encode(self, z0, N=None):
    traj = self.sample_ode(z0, N)
    z1 = traj[-1]
    return z1, 0, 0

class RectifiedFlow(BaseFlow):
  def get_train_tuple(self, z0=None, z1=None, t = None, eps = 1e-5):
    if t is None:
      t = torch.rand((z1.shape[0], 1), device=self.device)
    t = t * (1 - eps) + eps
    if len(z1.shape) == 2:
      z_t =  t * z1 + (1.-t) * z0
    elif len(z1.shape) == 4:
      t = t.view(-1, 1, 1, 1)
      z_t =  t * z1 + (1.-t) * z0
      t = t.view(-1)
    else:
      raise NotImplementedError(f"get_train_tuple not implemented for {self.__class__.__name__}.")
    target = z1 - z0 
    return z_t, t, target
  
  def get_train_tuple_ddpm(self, z0=None, z1=None, t = None):
    a = 19.9
    b = 0.1
    if t is None:
      t = torch.rand((z1.shape[0], 1), device=self.device)
      eps = 1e-5
      t = t * (1 - eps) + eps
    alpha_t = alpha(t)
    if len(z1.shape) == 2:
      z_t =  torch.sqrt(1 - alpha_t ** 2) * z1 + alpha_t * z0
    elif len(z1.shape) == 4:
      raise NotImplementedError
      t = t.view(-1, 1, 1, 1)
      z_t =  t * z1 + (1.-t) * z0
    else:
      raise NotImplementedError(f"get_train_tuple not implemented for {self.__class__.__name__}.")
    target = d_1_minus_alpha_sq_dt(t) * z1 + dalpha_dt(t) * z0
    return z_t, t, target

class NonlinearFlow(BaseFlow):
    def __init__(self, device, model=None, model_forward = None, num_steps=1000):
        self.model = model # generative ODEs
        self.model_forward = model_forward # forward ODEs
        self.N = num_steps
        self.device = device
    
    def get_train_tuple(self, z0=None, z1=None):
        t = torch.rand((z1.shape[0], 1), device=self.device)
        z_t =  self.model_forward(data = z0, noise = z1, t = t)
        z_t_dt = self.model_forward(data = z0, noise = z1, t = t + 1e-5)
        target = (z_t_dt - z_t) / 1e-5
        # z_t =  t * z1 + (1.-t) * z0
        # target = z1 - z0 
            
        return z_t, t, target
    @torch.no_grad()
    def sample_ode(self, z1=None, N=None):
      ### NOTE: Use Euler method to sample from the learned flow
      if N is None:
        N = self.N    
      dt = 1./N
      traj = [] # to store the trajectory
      z = z1.detach().clone()
      batchsize = z.shape[0]
      
      traj.append(z.detach().clone())
      for i in range(N, 0, -1):
        t = torch.ones((batchsize,1), device=self.device) * i / N
        pred = self.model(z, z1, t)
        z = z.detach().clone() - pred * dt
        
        traj.append(z.detach().clone())

      return traj
    
    @torch.no_grad()
    def sample_ode_forward(self, z0=None, noise=None, N=None):
      ### NOTE: Use Euler method to sample from the learned flow
      if N is None:
        N = self.N    
      dt = 1./N
      traj = [] # to store the trajectory
      z = z0.detach().clone()
      batchsize = z.shape[0]
      
      traj.append(z.detach().clone())
      for i in range(N):
        t = torch.ones((batchsize,1), device=self.device) * i / N
        z = self.model_forward(data = z0, noise = noise, t = t)
        
        traj.append(z.detach().clone())

      return traj
    
class ResidualFlow(RectifiedFlow):
  def __init__(self, device, model_list=None, num_steps=1000):
    self.model_list = model_list
    self.model = model_list[-1]
    self.N = num_steps
    self.device = device

  def get_train_tuple(self, z0=None, z1=None, t = None, N=2):
    if t is None:
      t = torch.randint(0,self.N+1,(z1.shape[0],1),device=self.device).float()/self.N
    z1 = self.sample_residual_ode(z1=z1,N=N)
    if len(z1.shape) == 2:
      z_t =  t * z1 + (1.-t) * z0
    elif len(z1.shape) == 4:
      t = t.view(-1, 1, 1, 1)
      z_t =  t * z1 + (1.-t) * z0
    else:
      raise NotImplementedError(f"get_train_tuple not implemented for {self.__class__.__name__}.")
    target = z1 - z0 
    return z_t, t, target
  
  def get_eval_dis(self, z0=None, z1=None, t = None, res_number = -1 ,N=2):
    if t is None:
      t = torch.randint(0,self.N+1,(z1.shape[0],1),device=self.device).float()/self.N
    if res_number == -1:
      z1 = self.sample_residual_ode(z1=z1,N=N)
    else:
      tq = lambda x: x
      if N is None:
        N = self.N    
      dt = -1./ N
      z = z1.detach().clone()
      batchsize = z.shape[0]
      for i in range(res_number):
        for j in tq(reversed(range(1,N+1))):
          s_t = torch.ones((batchsize,1), device=self.device) * j / N
          if len(z1.shape) == 2:
            vt = self.model_list[i](z, s_t)
          elif len(z1.shape) == 4:
            vt = self.model_list[i](z, s_t.squeeze())
          z = z.detach().clone() + vt * dt
      z1 = z
    if len(z1.shape) == 2:
      z_t =  t * z1 + (1.-t) * z0
    elif len(z1.shape) == 4:
      z_t =  t.view(-1, 1, 1, 1) * z1 + (1.-t.view(-1, 1, 1, 1)) * z0
    else:
      raise NotImplementedError(f"get_train_tuple not implemented for {self.__class__.__name__}.")
    target = z1 - z0
    if res_number!=-1:
      pred = self.model_list[res_number](z_t,t)
    else:
      pred = self.model_list[-1](z_t,t)
    return pred - target
  
  @torch.no_grad()
  def sample_ode(self, z0=None, N=None):
    if N == None:
      N = self.N
    traj = [] # to store the trajectory
    z = z0.detach().clone()
    batchsize = z.shape[0]
    traj.append(z.detach().clone())
    for i in reversed(range(len(self.model_list))):
      dt = 1./N
      for j in range(N):
        t = torch.ones((batchsize,1), device=self.device)* j / N
        if len(z0.shape) == 2:
          pred = self.model_list[i](z, t)
        elif len(z0.shape) == 4:
          pred = self.model_list[i](z, t.squeeze())
        z = z.detach().clone() + pred * dt
        traj.append(z.detach().clone())
    return traj
  
  @torch.no_grad()
  def sample_ode_generative(self, z1=None, N=None, use_tqdm=True, solver = 'euler'):
    assert solver in ['euler', 'heun','dpm_solver']
    tq = tqdm if use_tqdm else lambda x: x
    if N is None:
      N = self.N    
    if solver == 'heun' and N % 2 == 0:
      raise ValueError("N must be odd when using Heun's method.")
    if solver == 'heun':
      N = (N + 1) // 2
    dt = -1./ N
    traj = [] # to store the trajectory
    x0hat_list = []
    z = z1.detach().clone()
    batchsize = z.shape[0]
    traj.append(z.detach().clone())
    for i in range(len(self.model_list)):
      for j in tq(reversed(range(1,N+1))):
        t = torch.ones((batchsize,1), device=self.device) * j / N
        t_next = torch.ones((batchsize,1), device=self.device) * (j-1) / N
        if len(z1.shape) == 2:
          if solver == 'heun':
            raise NotImplementedError("Heun's method not implemented for 2D data.")
          vt = self.model_list[i](z, t)
        elif len(z1.shape) == 4:
          vt = self.model_list[i](z, t.squeeze())
          if solver == 'heun' and j > 1:
            z_next = z.detach().clone() + vt * dt
            vt_next = self.model_list[i](z_next, t_next.squeeze())
            vt = (vt + vt_next) / 2
          x0hat = z - vt * t.view(-1,1,1,1)
          x0hat_list.append(x0hat)
        z = z.detach().clone() + vt * dt
        traj.append(z.detach().clone())
    return traj, x0hat_list

  @torch.no_grad()
  def sample_residual_ode(self, z1=None, N=None,end = -1):
    tq = lambda x: x
    if N is None:
      N = self.N    
    dt = -1./ N
    z = z1.detach().clone()
    batchsize = z.shape[0]
    if end == -1:
      end = len(self.model_list)
    for i in range(end):
      for j in tq(reversed(range(1,N+1))):
        t = torch.ones((batchsize,1), device=self.device) * j / N
        if len(z1.shape) == 2:
          vt = self.model_list[i](z, t)
        elif len(z1.shape) == 4:
          vt = self.model_list[i](z, t.squeeze())
        z = z.detach().clone() + vt * dt
    return z


  @torch.no_grad()
  def sample_ode_generative_bbox(self, z1=None, N=None, use_tqdm=True, solver = 'RK45'):
    dshape = z1.shape
    device = z1.device
    check_set = [1,2,4,8,16]
    assert N in check_set,"N must be one of 1, 2, 4, 8, 16."
    start = torch.ones(z1.shape[0]).to(z1.device).float()
    end =  torch.zeros(z1.shape[0]).to(z1.device).float()
    step = (end - start) / N
    def ode_func(t, x, i):
      vec_t = t
      vt = self.model_list[i](x, vec_t)
      return vt
    for i in range(len(self.model_list)):
      z1 = RK(functools.partial(ode_func,i=i),z1,start,step,N,method=solver)
    return z1

class ConsistencyFlow(RectifiedFlow):
  def __init__(self, device, ema_model,model, num_steps=1000,TN=16):
    self.ema_model = ema_model
    self.model = model
    self.N = num_steps
    self.TN = TN
    self.device = device

  def get_train_tuple(self, z0=None, z1=None, t = None,eps=1e-2):
    if t is None:
      t = torch.rand((z0.shape[0],)).to(z1.device).float()
      t[t<=(1/self.TN)]=1/self.TN
    if len(z1.shape) == 2:
      pre_z_t =  t * z1 + (1.-t) * z0
    elif len(z1.shape) == 4:
      t = t.view(-1, 1, 1, 1)
      pre_z_t =  t * z1 + (1.-t) * z0
      t = t.view(-1)
    else:
      raise NotImplementedError(f"get_train_tuple not implemented for {self.__class__.__name__}.")
    
    with torch.no_grad():
      now_z_t = pre_z_t - (1/self.TN)*self.ema_model(pre_z_t,t)
      now_t = t - (1/self.TN)
    
    pred_z_t = self.model(pre_z_t,t)
    with torch.no_grad():
      gt_z_t = self.ema_model(now_z_t,now_t)
    return pred_z_t,gt_z_t
  

class OnlineSlimFlow(RectifiedFlow):
  def __init__(self, device, model, ema_model, num_steps=1000,TN=16):
    self.ema_model = ema_model
    self.model = model
    self.N = num_steps
    self.TN = TN
    self.device = device

  def get_train_tuple_one_step(self, z0=None, z1=None, t = None,eps=1e-5):
    if t is None:
      t = torch.rand((z0.shape[0],)).to(z1.device).float()
      t = t*(1-eps)+eps
    if len(z1.shape) == 2:
      pre_z_t =  t * z1 + (1.-t) * z0
    elif len(z1.shape) == 4:
      t = t.view(-1, 1, 1, 1)
      pre_z_t =  t * z1 + (1.-t) * z0
      t = t.view(-1)
    else:
      raise NotImplementedError(f"z1.shape should be 2 or 4.")
    
    with torch.no_grad():
      now_z_t = pre_z_t - (1/self.TN)*self.ema_model(pre_z_t,t)
      now_t = t - (1/self.TN)
      now_t = now_t.clamp(0,1)
    
    pred_z_t = self.model(pre_z_t,t)
    with torch.no_grad():
      ema_z_t = self.ema_model(now_z_t,now_t)
    gt_z_t = z1 - z0 
    return pred_z_t, ema_z_t, gt_z_t
  
  def get_train_tuple(self, z0=None, z1=None, t=None, eps=0.00001,pred_step=1):
    if pred_step==1:
      return self.get_train_tuple_one_step(z0,z1,t,eps)
    else:
      raise NotImplementedError(f"get_train_tuple not implemented for {self.__class__.__name__}.")
