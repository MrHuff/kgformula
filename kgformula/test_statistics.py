import torch
import gpytorch
from torch.distributions import Normal,StudentT,Bernoulli,Beta,Uniform,Exponential
import numpy as np
from pykeops.torch import LazyTensor,Genred
import time
from torch.utils.data import Dataset
from sklearn import metrics
import torch.nn as nn
try:
    from torch.cuda.amp import GradScaler,autocast
except Exception as e:
    print(e)
    print('Install nightly pytorch for mixed precision')

class HSIC_independence_test():
    def __init__(self,X,Y,n_samples):
        self.X = X
        self.Y = Y
        self.n = self.X.shape[0]
        if self.n>5000:
            idx = torch.randperm(2500)
            self.X = self.X[idx,:]
            self.Y = self.Y[idx,:]
            self.n = 2500
        self.kernel_base = gpytorch.kernels.Kernel()
        self.kernel_ls_init('ker_X',self.X)
        self.kernel_ls_init('ker_Y',self.Y)
        self.ker_X_eval = self.ker_X(self.X).evaluate()
        self.ker_Y_eval = self.ker_Y(self.Y).evaluate()
        self.ker_X_eval_ones = (self.ker_X_eval@torch.ones(*(self.n,1),device=self.X.device)).repeat(1,self.n)
        self.ker_Y_eval_ones = (self.ker_Y_eval@torch.ones(*(self.n,1),device=self.X.device)).repeat(1,self.n)
        self.X_cache = self.ker_X_eval-self.ker_X_eval_ones
        with torch.no_grad():
            self.ref_metric = torch.mean(self.X_cache*(self.ker_Y_eval-self.ker_Y_eval_ones)).item()
            self.p_val = self.calc_permute(self.X,self.Y,n_samples=n_samples)

    def get_median_ls(self,X):
        with torch.no_grad():
            if self.n>5000:
                idx = torch.randperm(2500)
                X = X[idx,:]
            d = self.kernel_base.covar_dist(x1=X,x2=X)
            ret = torch.sqrt(torch.median(d[d > 0]))
            return ret

    @staticmethod
    def calculate_pval(bootstrapped_list, test_statistic):
        pval = 1 - 1 / (bootstrapped_list.shape[0] + 1) * (1 + (bootstrapped_list <= test_statistic).sum())
        return pval

    def kernel_ls_init(self,name,data,ls=None):
        ker = gpytorch.kernels.RBFKernel()
        if ls is None:
            ls = self.get_median_ls(data)
        ker._set_lengthscale(ls)
        ker = ker.to(data.device)
        setattr(self,name,ker)

    def calculate_HSIC(self,idx):
        Y_ker = self.ker_Y_eval[idx,:]
        Y_ker = Y_ker[:,idx]
        return torch.mean(self.X_cache*(Y_ker-self.ker_Y_eval_ones[idx,:]))

    def calc_permute(self,X,Y,n_samples):
        self.bootstrapped=[]
        with torch.no_grad():
            for i in range(n_samples):
                idx = torch.randperm(self.X.shape[0])
                self.bootstrapped.append(self.calculate_HSIC(idx).item())
        self.bootstrapped = np.array(self.bootstrapped)
        return self.calculate_pval(self.bootstrapped,self.ref_metric)

def hsic_test(X,Y,n_sample = 250):
    test = HSIC_independence_test(X,Y,n_sample)
    return test.p_val

class Log1PlusExp(torch.autograd.Function):
    """Implementation of x ↦ log(1 + exp(x))."""
    @staticmethod
    def forward(ctx, x):
        exp = x.exp()
        ctx.save_for_backward(x)
        return x.where(torch.isinf(exp), exp.log1p())
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output / (1 + (-x).exp())

log_1_plus_exp = Log1PlusExp.apply

def NCE_objective(true_preds,fake_preds,kappa):
    _err = -torch.log(nu_sigmoid(true_preds,kappa)) - (1.-nu_sigmoid(fake_preds,kappa)).log().sum(dim=1)
    return _err.mean()

def NCE_objective_stable(true_preds,fake_preds):
    _err = -(true_preds-log_1_plus_exp(true_preds)-log_1_plus_exp(fake_preds).mean(dim=1))
    return _err.mean()

def accuracy_check(y_pred,Y):
    with torch.no_grad():
        y_pred = (y_pred.float() > 0.5).cpu().float().numpy()
        return np.mean(Y.cpu().numpy()==y_pred)

def auc_check(y_pred,Y):
    with torch.no_grad():
        y_pred = (y_pred.float() > 0.5).cpu().float().numpy()
        fpr, tpr, thresholds = metrics.roc_curve(Y.cpu().numpy(), y_pred, pos_label=1)
        auc =  metrics.auc(fpr, tpr)
        return auc

def nu_sigmoid(x,kappa):
    return 1./(1+kappa*torch.exp(-x))

def hsic_sanity_check_w(w,x,z,n_perms=250):
    _w = w.cpu().squeeze().numpy()
    idx_HSIC = np.random.choice(np.arange(x.shape[0]), x.shape[0], p=_w / _w.sum())
    p_val = hsic_test(x[idx_HSIC, :], z[idx_HSIC, :], n_perms)
    return p_val

class keops_RBFkernel(torch.nn.Module):
    def __init__(self,ls,x,y=None,device_id=0):
        super(keops_RBFkernel, self).__init__()
        self.device_id = device_id
        self.raw_lengthscale = torch.nn.Parameter(ls,requires_grad=False).contiguous()
        self.raw_lengthscale.requires_grad = False
        self.register_buffer('x', x.contiguous())
        self.shape = (x.shape[0],x.shape[0])
        if y is not None:
            self.register_buffer('y',y.contiguous())
        else:
            self.y = x
        self.gen_formula = None

    def get_formula(self,D,ls_size):
        aliases = ['G_0 = Pm(0, ' + str(ls_size) + ')',
                   'X_0 = Vi(1, ' + str(D) + ')',
                   'Y_0 = Vj(2, ' + str(D) + ')',
                   ]
        formula = 'Exp(-G_0*SqNorm2(X_0 - Y_0))'
        return formula,aliases

    def forward(self):
        if self.gen_formula is None:
            self.formula, self.aliases = self.get_formula(D=self.x.shape[1], ls_size=self.raw_lengthscale.shape[0])
            self.gen_formula = Genred(self.formula, self.aliases, reduction_op='Sum', axis=1, dtype='float32')
        return self.gen_formula(*[self.raw_lengthscale,self.x,self.y],backend='GPU',device_id=self.device_id)

def get_i_not_j_indices(n):
    vec = np.arange(0, n)
    vec_2 = np.arange(0, int(n ** 2), n+1)
    list_np = np.array(np.meshgrid(vec, vec)).T.reshape(-1, 2)
    list_np = np.delete(list_np, vec_2, axis=0)
    return list_np

class MLP(torch.nn.Module):
    def __init__(self,d,f=12,k=2,o=1):
        super(MLP, self).__init__()
        self.model = nn.ModuleList()
        self.model.append(nn.Linear(d, f))
        self.model.append(nn.Sigmoid())
        for i in range(k):
            self.model.append(nn.Linear(f, f))
            self.model.append(nn.Sigmoid())
        self.model.append(nn.Linear(f, o))

    def forward(self,X,Z):
        x = torch.cat([X,Z],dim=1)
        for l in self.model:
            x = l(x)
        return x

    def forward_predict(self,X,Z):
        return self.forward(X,Z)

    def get_w(self, x, y):
        return torch.exp(-self.forward(x,y))

class logistic_regression(torch.nn.Module):
    def __init__(self,d):
        super(logistic_regression, self).__init__()
        self.W = torch.nn.Linear(in_features=d,  out_features=1,bias=True)

    def forward(self,x,z):
        X = torch.cat([x,z],dim=1)
        return self.W(X)

    def forward_predict(self,X,Z):
        return self.forward(X,Z)

    def get_w(self, x, y):
        return torch.exp(-self.forward(x,y))

class classification_dataset(Dataset):
    def __init__(self,X,Z,bs=1.0,kappa=1,val_rate = 0.01):
        super(classification_dataset, self).__init__()
        self.n=X.shape[0]
        mask = np.array([False] * self.n)
        mask[0:round(val_rate*self.n)] = True
        np.random.shuffle(mask)
        self.X_train = X[~mask,:]
        self.Z_train = Z[~mask,:]
        self.X_val = X[mask]
        self.Z_val = Z[mask]
        self.bs_perc = bs
        self.device = X.device
        self.kappa = kappa
        self.train_mode()

    def train_mode(self):
        self.X = self.X_train
        self.Z = self.Z_train
        self.sample_indices_base = np.arange(self.X.shape[0])
        self.bs = int(round(self.bs_perc*self.X.shape[0]))

    def val_mode(self):
        self.X = self.X_val
        self.Z = self.Z_val
        self.sample_indices_base = np.arange(self.X.shape[0])
        self.bs = self.X.shape[0]

    def build_sampling_set(self,true_indices):
        np_cat = []
        for x in np.nditer(true_indices):
            np_cat.append(np.delete(self.sample_indices_base,x)[None,:])
        return np.concatenate(np_cat,axis=0)

    def sample_no_replace(self,fake_set,kappa,replace=False):
        np_cat = []
        for row in fake_set:
            np_cat.append(np.random.choice(row,kappa,replace))
        return np.concatenate(np_cat)

    def get_indices(self):
        if self.bs ==self.X.shape[0]:
            true_indices = self.sample_indices_base
        else:
            i_s = np.random.randint(0, self.X.shape[0] - 1 - self.bs)
            true_indices = np.arange(i_s,i_s+self.bs)
        fake_set = self.build_sampling_set(true_indices)
        fake_indices = self.sample_no_replace(fake_set,self.kappa,False)
        return true_indices,fake_indices#,HSIC_ref_indices_true,HSIC_ref_indices_fake

    def get_sample(self):
        T,F = self.get_indices()
        return self.X[T,:],self.Z[T,:],self.X[T.repeat(self.kappa),:],self.Z[F,:]

class density_estimator():
    def __init__(self, x, z, est_params=None, cuda=False, device=0, type='linear'):
        self.failed = False
        self.x = x
        self.z = z
        self.cuda = cuda
        self.n = self.x.shape[0]
        self.device = device
        self.est_params = est_params
        self.type = type
        self.kernel_base = gpytorch.kernels.Kernel()
        if type=='linear':
            self.alpha = est_params['alpha']
            self.diag = est_params['reg_lambda']*torch.eye(self.n)
            if self.cuda:
                self.diag = self.diag.cuda(self.device)
            self.linear_x_of_z()
            self.get_w_kdre()

        elif type=='gp':
            self.alpha = est_params['alpha']
            self.diag = est_params['reg_lambda']*torch.eye(self.n)
            if self.cuda:
                self.diag = self.diag.cuda(self.device)
            self.kernel_ls_init('kernel_tmp', self.z)
            self.gp_x_of_z()
            self.get_w_kdre()

        elif type=='semi':
            self.semi_cheat_x_of_z()

        elif type == 'kmm':
            self.diag = est_params['reg_lambda']*torch.eye(self.n)
            if self.cuda:
                self.diag = self.diag.cuda(self.device)
            self.kernel_mean_matching()

        elif type == 'classifier':
            dataset = self.create_classification_data()
            self.model = MLP(d=dataset.X.shape[1]+dataset.Z.shape[1],f=self.est_params['width'],k=self.est_params['layers']).to(self.x.device)
            self.w = self.train_classifier(dataset)

        elif type == 'linear_classifier':
            dataset = self.create_classification_data()
            self.model = logistic_regression(d=dataset.X.shape[1]+dataset.Z.shape[1]).to(self.x.device)
            self.w = self.train_classifier(dataset)

        elif type == 'random_uniform':
            self.w = torch.rand(*(self.x.shape[0],1),device=self.device)*2

    def retrain(self,x,z):
        self.x = x
        self.z = z
        if self.type == 'linear_classifier':
            dataset = self.create_classification_data()
            self.w = self.train_classifier(dataset)
        elif self.type == 'classifier':
            dataset = self.create_classification_data()
            self.w = self.train_classifier(dataset)

    def forward_pred(self,X,Z,):
        pred_T = self.model.forward_predict(X,Z)
        return pred_T.squeeze()

    def forward_func(self,X,Z,X_fake,Z_fake,loss_func,kappa):
        pred_T = self.model(X,Z)
        pred_F = self.model(X_fake,Z_fake)
        pred_F = pred_F.view(pred_T.shape[0],-1)
        return loss_func(pred_T,pred_F,kappa)

    def train_classifier(self,dataset):
        loss_func = NCE_objective
        opt = torch.optim.Adam(self.model.parameters(),lr=self.est_params['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.5, patience=1)
        if self.est_params['mixed']:
            scaler = GradScaler()
        counter = 0
        best = np.inf
        idx = torch.randperm(dataset.X_val.shape[0])
        kappa = self.est_params['kappa']
        for i in range(self.est_params['max_its']):
            X_true,Z_true,X_fake,Z_fake= dataset.get_sample()
            opt.zero_grad()
            if self.est_params['mixed']:
                with autocast():
                    l = self.forward_func(X_true,Z_true,X_fake,Z_fake,loss_func,kappa)
                scaler.scale(l).backward()
                scaler.step(opt)
                scaler.update()
            else:
                l = self.forward_func(X_true, Z_true, X_fake, Z_fake,loss_func,kappa)
                l.backward()
                opt.step()

            if i%(self.est_params['max_its']//25)==0:
                # print(l)
                with torch.no_grad():
                    dataset.val_mode()
                    pred_T = self.forward_pred(dataset.X,dataset.Z)
                    pred_F = self.forward_pred(dataset.X,dataset.Z[idx])
                    pred_F = pred_F.view(pred_T.shape[0], -1)
                    logloss = loss_func(pred_T,pred_F,kappa)
                    print(f'logloss epoch {i}: {logloss}')
                    scheduler.step(logloss)
                    if logloss.item()<best:
                        best = logloss.item()
                        counter=0
                    else:
                        counter+=1
                    dataset.train_mode()
            if counter>self.est_params['kill_counter']:
                print('stopped improving, stopping')
                break
        with torch.no_grad():
            w = self.model.get_w(self.x,self.z)
            _w = w.cpu().squeeze().numpy()
            idx_HSIC = np.random.choice(np.arange(self.x.shape[0]),self.x.shape[0],p=_w/_w.sum())
            p_val = hsic_test(self.x[idx_HSIC,:],self.z[idx_HSIC,:],self.est_params['n_sample'])
            print(f'HSIC_pval : {p_val}')
            self.hsic_pval = p_val
            if p_val<self.est_params['criteria_limit']:
                self.failed=True
                print('failed')

        return w

    def create_classification_data(self):
        with torch.no_grad():
            self.kappa = self.est_params['kappa']
        return classification_dataset(self.x,
                                      self.z,
                                      bs=self.est_params['bs_ratio'],
                                      kappa=self.kappa,
                                      val_rate=self.est_params['val_rate'])

    def kernel_mean_matching(self):
        with torch.no_grad():
            list_idx = torch.from_numpy(get_i_not_j_indices(self.n)) #Seems to be working alright!
            torch_idx_x,torch_idx_z = list_idx.unbind(dim=1)
            data_extended = torch.cat([self.x[torch_idx_x],self.z[torch_idx_z]],dim=1)
            data = torch.cat([self.x,self.z],dim=1)
            self.kernel_ls_init('kernel_up', data)
            ls = self.get_median_ls_XY(data,data_extended)
            y = self.kernel_ls_init_keops(ls=ls,data=data,data_2=data_extended)
            self.w,_ = torch.solve(y,(self.kernel_up+self.diag))

    def get_w_kdre(self):
        self.kernel_ls_init('kernel_up', self.x)
        self.kernel_ls_init('kernel_down', self.x, self.down_estimator)
        with torch.no_grad():
            self.kernel_down = self.kernel_down.evaluate()
            self.h_hat = self.kernel_up.mean(dim=1,keepdim=True)
            self.H = self.alpha/self.n * self.kernel_up@self.kernel_up + (1-self.alpha)/self.n * (self.kernel_down@self.kernel_down) + self.diag
            self.theta,_ = torch.solve(self.h_hat, self.H)
            self.w = self.kernel_up@self.theta

    def return_weights(self):
        return self.w.squeeze()

    def linear_x_of_z(self):
        down = torch.cat([self.z, torch.ones_like(self.z)], dim=1)
        with torch.no_grad():
            self.down_estimator = down@(torch.inverse(down.t()@down) @ (down.t() @ self.x))

    def semi_cheat_x_of_z(self):
        with torch.no_grad():
            self.linear_x_of_z()
            res = self.z - self.down_estimator
            p_1 = Normal(0,scale=res.var()**0.5)
            p_2 = Normal(0, scale=self.z.var() ** 0.5)
            self.w = (p_2.log_prob(self.z - self.z.mean()) - p_1.log_prob(res)).exp()

    def gp_x_of_z(self):
        with torch.no_grad():
            s,_ = torch.solve(self.x, self.kernel_tmp + self.diag)
            self.down_estimator = self.kernel_tmp@s

    def get_median_ls_XY(self,X,Y):
        with torch.no_grad():
            if X.shape[0]>5000:
                X = X[torch.randperm(5000),:]
            if Y.shape[0]>5000:
                Y = Y[torch.randperm(5000),:]
            d = self.kernel_base.covar_dist(x1=X,x2=Y)
            ret = torch.sqrt(torch.median(d[d > 0]))
            return ret

    def get_median_ls(self,X):
        with torch.no_grad():
            d = self.kernel_base.covar_dist(x1=X,x2=X)
            ret = torch.sqrt(torch.median(d[d > 0]))
            return ret

    def kernel_ls_init_keops(self,ls,data,data_2=None):
        with torch.no_grad():
            ker = keops_RBFkernel(ls=1/ls.unsqueeze(0),x=data,y=data_2,device_id=self.device)
            return ker()/(self.n-1)

    def kernel_ls_init(self,name,data,data_2=None,ls=None):
        ker = gpytorch.kernels.RBFKernel()
        if ls is None:
            ls = self.get_median_ls(data)
        ker._set_lengthscale(ls)
        if self.cuda:
            ker = ker.cuda(self.device)
            if data_2 is None:
                setattr(self,name,ker(data).evaluate())
            else:
                setattr(self,name,ker(data,data_2))
        return ls

class consistent_weighted_HSIC():
    def __init__(self,X,Y,w,Z=None,cuda=False,device=0,half_mode=False):
        self.X = X
        self.Y = Y
        self.n = X.shape[0]
        self.w = w.unsqueeze(-1)
        self.W = w@w.t()
        self.cuda = cuda
        self.device = device
        self.half_mode = half_mode
        with torch.no_grad():
            self.kernel_base = gpytorch.kernels.Kernel().cuda(device)
            for name, data in zip(['X', 'Y'], [self.X, self.Y]):
                self.kernel_ls_init(name, data)

            # self.cache_W_K = self.kernel_X*self.W/self.n**2
            # self.cache_sum_K = self.kernel_X.sum()/self.n**2
            # self.cache_K_1 = self.kernel_X @ torch.ones_like(self.w)
            #
            # self.cache_W_L = self.kernel_Y*self.W/self.n**2
            # self.cache_L_w = self.kernel_Y@self.w
            # self.middle_term  = self.cache_W_L.sum() * self.cache_sum_K
            # self.Y_center_1 =self.cache_W_K - (self.cache_W_L@torch.ones_like(self.w)/self.n).repeat(1,self.n)
            # self.Y_center_2 = self.Y_center_1 - (self.Y_center_1@torch.ones_like(self.w)/self.n).repeat(1,self.n)

            self.H_w = torch.diag(self.w)/self.n-self.w.repeat(1,self.n)/self.n**2
            self.H_wXH_w = self.H_w@(self.kernel_X@self.H_w.t())
            self.H_wYH_w = self.H_w@(self.kernel_Y@self.H_w.t())
            #
            # self.H_w1 = torch.diag(self.w)/self.n-torch.ones(*(self.n,self.n),device=self.device)/self.n**2
            # self.H_w1XH_w1 = self.H_w1@(self.kernel_X@self.H_w1.t())
            # self.H_w1YH_w1 = self.H_w1@(self.kernel_Y@self.H_w1.t())

    def get_permuted_Y_kernel(self):
        idx = torch.randperm(self.n)
        kernel_Y = self.kernel_Y[:,idx]
        kernel_Y = kernel_Y[idx,:]
        return kernel_Y,idx

    def get_permuted_X_kernel(self):
        idx = torch.randperm(self.n)
        kernel_X = self.kernel_X[:,idx]
        kernel_X = kernel_X[idx,:]
        return kernel_X,idx

    def calculate_weighted_statistic(self):
        with torch.no_grad():
            # return self.n * torch.sum(self.kernel_Y*self.H_w1XH_w1)
            #
            return self.n * torch.sum(self.kernel_Y*self.H_wXH_w)
            # return (torch.sum(self.cache_W_K*self.kernel_Y)+self.middle_term-2*(self.w*self.cache_K_1 * self.cache_L_w).sum()/self.n**3)*self.n

    def permute_Y_sanity(self):
        kernel_Y,_ = self.get_permuted_Y_kernel()
        return self.n*torch.sum(kernel_Y*self.H_wXH_w)
    #
    # def permute_Y_sanity_2(self):
    #     kernel_Y,_ = self.get_permuted_Y_kernel()
    #     return self.n*torch.sum(kernel_Y*self.H_w1XH_w1)


    # def permute_X_sanity(self):
    #     kernel_X,idx = self.get_permuted_X_kernel()
    #     return self.n*torch.sum(self.H_wYH_w*kernel_X)

    # def permute_Y(self):
    #     kernel_Y,_ = self.get_permuted_Y_kernel()
    #     return (torch.sum(self.cache_W_K*kernel_Y)+self.cache_sum_K*(self.W*kernel_Y).sum()/self.n**2-2*(self.w*self.cache_K_1 * (kernel_Y@self.w)).sum()/self.n**3)*self.n

    # def permute_X(self):
    #     kernel_X,idx = self.get_permuted_X_kernel()
    #     return (torch.sum(self.cache_W_L * kernel_X) + self.middle_term - 2 * (
    #                 self.w * self.cache_K_1[idx, :] * self.cache_L_w).sum()) * self.n
    #
    # def permute_X_2(self):
    #     kernel_X,idx = self.get_permuted_X_kernel()
    #     return self.n*torch.sum(kernel_X*self.Y_center_2)

    def permutation_calculate_weighted_statistic(self):
        with torch.no_grad():
            return self.permute_Y_sanity()

    def kernel_ls_init(self, name, data):
        setattr(self, f'ker_obj_{name}',
                gpytorch.kernels.RBFKernel().cuda(self.device) if self.cuda else gpytorch.kernels.RBFKernel())
        ls = self.get_median_ls(data)
        getattr(self, f'ker_obj_{name}')._set_lengthscale(ls)
        if self.half_mode:
            setattr(self, f'kernel_{name}', getattr(self, f'ker_obj_{name}')(data).evaluate().half())
        else:
            setattr(self, f'kernel_{name}', getattr(self, f'ker_obj_{name}')(data).evaluate())

    def get_median_ls(self, X):
        with torch.no_grad():
            d = self.kernel_base.covar_dist(x1=X, x2=X)
            ret = torch.sqrt(torch.median(d[d > 0]))
            return ret

class weighted_stat(): #HAPPY MISTAKE?!?!??!?!?!?!?!?
    def __init__(self,X,Y,Z,w,do_null = True,cuda=False,device=0,half_mode=False):
        with torch.no_grad():
            self.device = device
            self.cuda = cuda
            self.n = X.shape[0]
            self.one_n_1 = torch.ones(*(self.n,1))
            self.half_mode = half_mode
            if cuda:
                self.one_n_1 = self.one_n_1.cuda(device)
            else:
                self.device = 'cpu'
            self.X = X
            self.Y = Y
            self.Z = Z
            self.w = w.unsqueeze(-1)
            self.W = self.w@self.w.t()
            self.W = self.W/self.n**2
            self.do_null=do_null
            self.kernel_base = gpytorch.kernels.Kernel().cuda(self.device)
            for name,data in zip(['X','Y'],[self.X,self.Y]):
                self.kernel_ls_init(name,data)
            if self.half_mode:
                self.H = self.H.half()
                self.w = self.w.half()
                self.W = self.W.half()

    def kernel_ls_init(self,name,data):
        setattr(self, f'ker_obj_{name}', gpytorch.kernels.RBFKernel().cuda(self.device) if self.cuda else gpytorch.kernels.RBFKernel())
        ls = self.get_median_ls(data)
        getattr(self,f'ker_obj_{name}')._set_lengthscale(ls)
        if self.half_mode:
            setattr(self,f'kernel_{name}',getattr(self,f'ker_obj_{name}')(data).evaluate().half())
        else:
            setattr(self,f'kernel_{name}',getattr(self,f'ker_obj_{name}')(data).evaluate())

    def get_median_ls(self,X):
        with torch.no_grad():
            d = self.kernel_base.covar_dist(x1=X,x2=X)
            ret = torch.sqrt(torch.median(d[d > 0]))
            return ret

class weighted_statistic_new(weighted_stat):
    def __init__(self,X,Y,Z,w,do_null = True,cuda=False,device=0):
        super(weighted_statistic_new, self).__init__(X, Y, Z, w, do_null, cuda, device)
        with torch.no_grad():
            self.sum_mean_X = self.kernel_X.mean()
            self.X_ker_n_1 = self.kernel_X @ self.one_n_1 / self.n
            self.X_ker_ones = self.X_ker_n_1.repeat(1,self.n)
            self.X_ker_H_2= self.kernel_X - self.X_ker_ones * 2

            self.sum_mean_Y = self.kernel_Y.mean()
            self.Y_ker_n_1 = self.kernel_Y @ self.one_n_1 / self.n
            self.Y_ker_ones =self.Y_ker_n_1.repeat(1,self.n)
            self.Y_ker_H_2= self.kernel_Y - 2 * self.Y_ker_ones

            self.term_1 = self.W*self.X_ker_H_2*self.n
            self.term_2 = 2*self.W*self.kernel_X*self.n
            self.term_3 = 2*self.W*self.X_ker_ones*self.n
            self.term_4 = 2*self.W*self.n
            self.term_5 = self.W*self.sum_mean_X*self.n
            self.term_6 = (self.W*self.sum_mean_Y*self.X_ker_H_2).sum()*self.n
            self.term_7 = (self.W*self.sum_mean_Y*self.sum_mean_X).sum()*self.n

    def permutation_calculate_weighted_statistic(self):
        with torch.no_grad():
            idx = torch.randperm(self.n)
            Y_ker = self.kernel_Y[idx, :]
            Y_ker = Y_ker[:,idx]
            Y_ker_n_1 = self.Y_ker_n_1[idx,:]
            Y_ker_ones =Y_ker_n_1.repeat(1,self.n)
            Y_ker_H_2 = Y_ker - 2*Y_ker_ones
            test_stat = self.term_1 * Y_ker - self.term_2 * Y_ker_ones + self.term_3 * Y_ker_ones + self.term_4 * (
                        self.X_ker_n_1 @ Y_ker_n_1.t()) + self.term_5 * Y_ker_H_2
            return test_stat.sum() + self.term_6 + self.term_7

    def calculate_weighted_statistic(self):
        with torch.no_grad():
            test_stat = self.term_1 * self.kernel_Y - self.term_2 * self.Y_ker_ones + self.term_3 * self.Y_ker_ones + self.term_4 * (
                    self.X_ker_n_1 @ self.Y_ker_n_1.t()) + self.term_5 * self.Y_ker_H_2
            return test_stat.sum() + self.term_6 + self.term_7






