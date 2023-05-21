import torch,random
from tqdm import tqdm
from utils import alpha, dalpha_dt, d_1_minus_alpha_sq_dt
from scipy import integrate
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

  
  def sample_ode_generative_bbox(self, z1=None, N=None, use_tqdm=True, solver = 'RK45', eps = 1e-3, rtol=1e-5, atol=1e-5,if_pred_x0=False):
    dshape = z1.shape
    device = z1.device
    def ode_func(t, x):
      x = torch.from_numpy(x.reshape(dshape)).to(device).type(torch.float32)
      vec_t = torch.ones(dshape[0], device=x.device) * t
      if if_pred_x0:
        vt = z1 - self.model(x,vec_t)
      else:
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
    

class ConsistencyFlow(RectifiedFlow):
  def __init__(self, device,model, ema_model,num_steps=1000,TN=16,discrete=False):
    self.ema_model = ema_model
    import copy
    self.pre_train_model = copy.deepcopy(model.module)
    self.discrete = discrete
    self.model = model
    self.N = num_steps
    self.TN = TN
    self.device = device

  def get_train_tuple(self, z0=None, z1=None, t = None,eps=1e-3):
    if t is None:
      if self.discrete:
        t = torch.randint(1,self.TN+1,(z0.shape[0],)).to(z1.device).float()/self.TN
      else:
        t = torch.rand((z0.shape[0],)).to(z1.device).float()
    if len(z1.shape) == 2:
      pre_z_t =  t * z1 + (1.-t) * z0
    elif len(z1.shape) == 4:
      t = t.view(-1, 1, 1, 1)
      pre_z_t =  t * z1 + (1.-t) * z0
      t = t.view(-1)
    else:
      raise NotImplementedError(f"get_train_tuple not implemented for {self.__class__.__name__}.")
    
    with torch.no_grad():
      now_z_t = pre_z_t - (1/self.TN)*self.pre_train_model(pre_z_t,t)
      now_t = torch.clamp(t - (1/self.TN),1/self.TN,1)
    
    pred_z_t = self.model(pre_z_t,t)
    with torch.no_grad():
      gt_z_t = self.ema_model(now_z_t,now_t)
    return pred_z_t, gt_z_t
  
class ProgDistFlow(RectifiedFlow):
  def __init__(self, device,student_model, teacher_model,TN=16):
    self.student_model = student_model
    self.teacher_model = teacher_model
    self.TN = TN
    self.device = device
  
  @property
  def model(self):
    return self.student_model
  
  def get_train_tuple(self, z0=None, z1=None, t = None,eps=1e-3):
    if t is None:
      t = torch.randint(2,self.TN+1,(z0.shape[0],)).to(z1.device).float()/self.TN
    if len(z1.shape) == 2:
      pre_z_t =  t * z1 + (1.-t) * z0
    elif len(z1.shape) == 4:
      t = t.view(-1, 1, 1, 1)
      pre_z_t =  t * z1 + (1.-t) * z0
      t = t.view(-1)
    else:
      raise NotImplementedError(f"get_train_tuple not implemented for {self.__class__.__name__}.")
    student_output = self.student_model(pre_z_t,t)
    with torch.no_grad():
      t_h = torch.clamp(t - (1/self.TN),1/self.TN,1)
      pre_z_t_h = pre_z_t - (1/self.TN)*self.teacher_model(pre_z_t,t)
      pre_z_t_2h = pre_z_t_h - (1/self.TN)*self.teacher_model(pre_z_t_h,t_h)
      teacher_output = (pre_z_t - pre_z_t_2h)/(2/self.TN)
    return student_output,teacher_output




class CatchUpFlow(RectifiedFlow):
  def __init__(self, device, model, ema_model, generator_list=None, num_steps=1000,TN=16,adapt_cu="origin",add_prior_z=False,discrete=False):
    self.ema_model = ema_model
    self.model = model
    self.N = num_steps
    assert adapt_cu in ["origin","rule","uniform"],"adapt_cu must be one of 'origin','rule','uniform'."
    self.adapt_cu = adapt_cu
    self.add_prior_z = add_prior_z
    self.discrete = discrete
    self.TN = TN
    self.device = device
    if generator_list == None:
      self.cu_number = 1
    else:
      assert isinstance(generator_list,list)
      self.cu_number = len(generator_list) + 1
    self.generator_list = generator_list # include [g2,g3]


  def get_train_tuple_one_step(self, z0=None, z1=None, t = None,eps=1e-5):
    ori_z = z1 if self.add_prior_z else None
    if t is None:
      if self.discrete:
        t = torch.randint(1,self.TN+1,(z0.shape[0],)).to(z1.device).float()/self.TN
      else:
        t = torch.rand((z0.shape[0],)).to(z1.device).float()
        t = t*(1-eps)+eps
    
    if self.discrete:
      if self.adapt_cu=="uniform":
          dt = (torch.rand((z0.shape[0],)).to(z1.device).float() * t *self.TN).int().float()/self.TN
      else:
          dt = torch.ones((z0.shape[0],)).to(z1.device).float()/self.TN
      mask = (t>=dt)
    else:
      if self.adapt_cu=="uniform":
        dt = torch.rand((z0.shape[0],)).to(z1.device).float() * (1/self.TN)
      elif self.adapt_cu=="rule":
        dt = t * (1/self.TN)
      else:
        dt = torch.ones((z0.shape[0],)).to(z1.device).float() * (1/self.TN)
      mask = (t>=(eps+dt))
    if len(z1.shape) == 2:
      pre_z_t =  t * z1 + (1.-t) * z0
    elif len(z1.shape) == 4:
      t = t.view(-1, 1, 1, 1)
      pre_z_t =  t * z1 + (1.-t) * z0
      t = t.view(-1)
    else:
      raise NotImplementedError(f"z1.shape should be 2 or 4.")
    
    ############################################## Train Model Prediction ###############################################
    pred_z_t = self.model(pre_z_t,(t*self.TN).int() if self.discrete else t,ori_z=ori_z)

    ############################################## EMA Model Prediction #################################################
    with torch.no_grad():
      now_z_t = pre_z_t - dt.view(dt.shape[0],1,1,1)*self.model(pre_z_t,(t*self.TN).int() if self.discrete else t,ori_z=ori_z)
      now_t = t - dt
      if self.discrete:
        now_t = now_t.clamp(1/self.TN,1)
      else:
        now_t = now_t.clamp(eps,1)
      ema_z_t = self.model(now_z_t,(now_t*self.TN).int() if self.discrete else now_t,ori_z=ori_z)
      ema_z_t[~mask] = (z1-z0)[~mask]
    ############################################## Ground Truth #########################################################
    gt_z_t = z1 - z0

    return pred_z_t, ema_z_t, gt_z_t


  def get_train_tuple_two_step(self, z0 = None, z1 = None, t = None,eps = 1e-5):
    ori_z = z1 if self.add_prior_z else None
    if t is None:
      if self.discrete:
        t = torch.randint(1,self.TN+1,(z0.shape[0],)).to(z1.device).float()/self.TN
      else:
        t = torch.rand((z0.shape[0],)).to(z1.device).float()
        t = t*(1-eps)+eps
    
    if self.discrete:
        dt = torch.ones((z0.shape[0],)).to(z1.device).float() * (1/self.TN)
        mask = (t>=2*dt)
    else:
      if self.adapt_cu=="uniform":
        dt = torch.rand((z0.shape[0],)).to(z1.device).float() * (1/self.TN)
      elif self.adapt_cu=="rule":
        dt = t * (1/self.TN)
      else:
        dt = torch.ones((z0.shape[0],)).to(z1.device).float() * (1/self.TN)
      mask = (t>=(eps+2*dt))

    if len(z1.shape) == 2:
      pre_z_t =  t * z1 + (1.-t) * z0
    elif len(z1.shape) == 4:
      t = t.view(-1, 1, 1, 1)
      pre_z_t =  t * z1 + (1.-t) * z0
      t = t.view(-1)
    else:
      raise NotImplementedError(f"z1.shape should be 2 or 4.")
    
    ############################################## Train Model Prediction ###############################################
    pred_1_z_t,features = self.model(pre_z_t,(t*self.TN).int() if self.discrete else t,return_features=True,ori_z=ori_z)
    pred_2_z_t = self.generator_list[0](features)
    pred_list = [pred_1_z_t,pred_2_z_t]

    ############################################## EMA Model Prediction #################################################
    with torch.no_grad():
      h = dt.view(dt.shape[0],1,1,1)
      k1 = self.model(pre_z_t,(t*self.TN).int() if self.discrete else t,ori_z=ori_z)
      k2 = self.model(pre_z_t-h*k1,(t*self.TN-1).int() if self.discrete else t-dt,ori_z=ori_z)
      ema_1_z_t = pre_z_t - h*((1/2)*k1+(1/2)*k2)
      ema_2_z_t = pre_z_t - h*(k1+k2)
      ema_1_z_t = self.model(ema_1_z_t,(t*self.TN-1).int().clamp(1,self.TN) if self.discrete else torch.clamp(t-dt,eps,1),ori_z=ori_z)
      ema_2_z_t = self.model(ema_2_z_t,(t*self.TN-2).int().clamp(1,self.TN) if self.discrete else torch.clamp(t-2*dt,eps,1),ori_z=ori_z)
      ema_1_z_t[~mask] = (z1-z0)[~mask]
      ema_2_z_t[~mask] = (z1-z0)[~mask]
      ema_list = [ema_1_z_t,ema_2_z_t]
    ############################################## GT ##################################################################
    gt_z_t = z1 - z0 

    return pred_list, ema_list, gt_z_t
  
  def get_train_tuple_three_step(self, z0=None, z1=None, t = None,eps=1e-5):
      ori_z = z1 if self.add_prior_z else None
      if t is None:
        if self.discrete:
          t = torch.randint(1,self.TN+1,(z0.shape[0],)).to(z1.device).float()/self.TN
        else:
          t = torch.rand((z0.shape[0],)).to(z1.device).float()
          t = t*(1-eps)+eps
      
      if self.discrete:
          dt = torch.ones((z0.shape[0],)).to(z1.device).float() * (1/self.TN)
          mask = (t>=3*dt)
      else:
        if self.adapt_cu=="uniform":
          dt = torch.rand((z0.shape[0],)).to(z1.device).float() * (1/self.TN)
        elif self.adapt_cu=="rule":
          dt = t * (1/self.TN)
        else:
          dt = torch.ones((z0.shape[0],)).to(z1.device).float() * (1/self.TN)
        mask = (t>=(eps+3*dt))
        
      if len(z1.shape) == 2:
        pre_z_t =  t * z1 + (1.-t) * z0
      elif len(z1.shape) == 4:
        t = t.view(-1, 1, 1, 1)
        pre_z_t =  t * z1 + (1.-t) * z0
        t = t.view(-1)
      else:
        raise NotImplementedError(f"z1.shape should be 2 or 4.")
      ############################################## Train Model Prediction ###############################################
      pred_1_z_t,features = self.model(pre_z_t,(t*self.TN).int() if self.discrete else t,return_features=True,ori_z=ori_z)
      pred_2_z_t = self.generator_list[0](features)
      pred_3_z_t = self.generator_list[1](features)
      pred_list = [pred_1_z_t,pred_2_z_t,pred_3_z_t]

      ############################################## EMA Model Prediction #################################################
      with torch.no_grad():
        h = dt.view(dt.shape[0],1,1,1)
        k1 = self.model(pre_z_t,(t*self.TN).int() if self.discrete else t,ori_z=ori_z)
        k2 = self.model(pre_z_t-h*k1,(t*self.TN-1).int().clamp(1,self.TN) if self.discrete else torch.clamp(t-dt,eps,1),ori_z=ori_z)
        k3 = self.model(pre_z_t-(7*h*k1/4+1*h*k2/4),(t*self.TN-2).int().clamp(1,self.TN) if self.discrete else torch.clamp(t-2*dt,eps,1),ori_z=ori_z)
        ema_1_z_t = pre_z_t - h*((5/12)*k1+(2/3)*k2-(1/12)*k3)
        ema_2_z_t = pre_z_t - 2*h*((5/12)*k1+(2/3)*k2-(1/12)*k3)
        ema_3_z_t = pre_z_t - 3*h*((5/12)*k1+(2/3)*k2-(1/12)*k3)
        ema_1_z_t = self.model(ema_1_z_t,(t*self.TN-1).int().clamp(1,self.TN) if self.discrete else torch.clamp(t-1*dt,eps,1),ori_z=ori_z)
        ema_2_z_t = self.model(ema_2_z_t,(t*self.TN-2).int().clamp(1,self.TN) if self.discrete else torch.clamp(t-2*dt,eps,1),ori_z=ori_z)
        ema_3_z_t = self.model(ema_3_z_t,(t*self.TN-3).int().clamp(1,self.TN) if self.discrete else torch.clamp(t-3*dt,eps,1),ori_z=ori_z)
        ema_1_z_t[~mask] = (z1-z0)[~mask]
        ema_2_z_t[~mask] = (z1-z0)[~mask]
        ema_3_z_t[~mask] = (z1-z0)[~mask]
        ema_list = [ema_1_z_t,ema_2_z_t,ema_3_z_t]

      ############################################## GT ##################################################################
      gt_z_t = z1 - z0 

      return pred_list, ema_list, gt_z_t
        


  def get_train_tuple(self, z0=None, z1=None, t=None, eps=0.00001,pred_step=1):
      if pred_step==1:
        return self.get_train_tuple_one_step(z0,z1,t,eps)
      elif pred_step==2:
        return self.get_train_tuple_two_step(z0,z1,t,eps)
      elif pred_step==3:
        return self.get_train_tuple_three_step(z0,z1,t,eps)
      else:
        raise NotImplementedError(f"get_train_tuple not implemented for {self.__class__.__name__}.")
  

  def get_eval_dis(self, z0=None, z1=None, t = None ,N=2):
    if t is None:
      t = torch.randint(0,self.N+1,(z1.shape[0],1),device=self.device).float()/self.N
    if N is None:
      N = self.N    
    if len(z1.shape) == 2:
      z_t =  t * z1 + (1.-t) * z0
    elif len(z1.shape) == 4:
      z_t =  t.view(-1, 1, 1, 1) * z1 + (1.-t.view(-1, 1, 1, 1)) * z0
    else:
      raise NotImplementedError(f"get_train_tuple not implemented for {self.__class__.__name__}.")
    target = z1 - z0
    pred = self.model(z_t,(t*self.TN).int() if self.discrete else t)
    return pred - target
  
  @torch.no_grad()
  def sample_ode_generative(self, z1=None, N=None, use_tqdm=True, solver = 'euler',momentum=0.0,generator_id=1):
    if generator_id == 1:
      model_fn = lambda z,t: self.model(z,t)
    else:
      model_fn = lambda z,t: self.generator_list[generator_id-2](self.model(z,t,return_features=True)[1])
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
    vt = 0.
    traj.append(z.detach().clone())
    for i in tq(reversed(range(1,N+1))):
      t = torch.ones((batchsize,1), device=self.device) * i / N
      t_next = torch.ones((batchsize,1), device=self.device) * (i-1) / N
      if len(z1.shape) == 2:
        if solver == 'heun':
          raise NotImplementedError("Heun's method not implemented for 2D data.")
        _vt = model_fn(z, t)
      elif len(z1.shape) == 4:
        _vt = model_fn(z, (t*self.TN).int().squeeze() if self.discrete else t.squeeze())
        if solver == 'heun':
          if i!=1:
            z_next = z.detach().clone() + _vt * dt
            vt_next = model_fn(z_next, (t_next*self.TN).int().squeeze() if self.discrete else t_next.squeeze())
            _vt = (_vt + vt_next) / 2
        if i==N:
          vt = _vt.detach().clone()
        else:
          vt = _vt.detach().clone() *(1-momentum) + momentum * vt
        x0hat = z - vt * t.view(-1,1,1,1)
        x0hat_list.append(x0hat)
      
      z = z.detach().clone() + vt * dt
      
      traj.append(z.detach().clone())
    return traj, x0hat_list
  
  @torch.no_grad()
  def sample_ode(self, z0=None, N=None):
    ### NOTE: Use Euler method to sample from the learned flow
    ori_z = torch.randn_like(z0).to(z0.device) if self.add_prior_z else None
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
        pred = self.model(z, (t*self.TN).int() if self.discrete else t ,ori_z=ori_z)
      elif len(z0.shape) == 4:
        pred = self.model(z, (t*self.TN).int().squeeze() if self.discrete else t.squeeze(),ori_z=ori_z)
      z = z.detach().clone() + pred * dt
      
      traj.append(z.detach().clone())
    return traj
