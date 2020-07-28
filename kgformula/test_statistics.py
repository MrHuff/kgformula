import torch
import gpytorch
from torch.distributions import Normal,StudentT,Bernoulli,Beta,Uniform,Exponential
import numpy as np
from pykeops.torch import LazyTensor,Genred
import time
from kgformula.networks import *
import os
try:
    from torch.cuda.amp import GradScaler,autocast
except Exception as e:
    print(e)
    print('Install nightly pytorch for mixed precision')
from cvxopt import matrix, solvers
import math

def kernel_mean_matching(K,kappa,nz, B=1.0, eps=None):
    if eps == None:
        eps = B / math.sqrt(nz)
    K = matrix(K)
    kappa = matrix(kappa)
    G = matrix(np.r_[np.ones((1, nz)), -np.ones((1, nz)), np.eye(nz), -np.eye(nz)])
    h = matrix(np.r_[nz * (1 + eps), nz * (eps - 1), B * np.ones((nz,)), np.zeros((nz,))])
    sol = solvers.qp(K, -kappa, G, h)
    coef = np.array(sol['x'])
    return coef


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
        self.tmp_path = f'./tmp_folder_{self.device}/'
        if not os.path.exists(self.tmp_path):
            os.makedirs(self.tmp_path)
        if type == 'kmm':
            self.diag = est_params['reg_lambda']*torch.eye(self.n)
            if self.cuda:
                self.diag = self.diag.cuda(self.device)
            self.w = self.kernel_mean_matching()

        elif type=='kmm_qp':
            self.diag = est_params['reg_lambda'] * torch.eye(self.n)
            if self.cuda:
                self.diag = self.diag.cuda(self.device)
            self.w = self.kernel_mean_matching_qp()

        elif type == 'NCE':

            dataset = self.create_classification_data()
            self.model = MLP(d=dataset.X.shape[1]+dataset.Z.shape[1],f=self.est_params['width'],k=self.est_params['layers']).to(self.x.device)
            self.w = self.train_classifier(dataset)

        elif type == 'linear_classifier':
            dataset = self.create_classification_data()
            self.model = logistic_regression(d=dataset.X.shape[1]+dataset.Z.shape[1]).to(self.x.device)
            self.w = self.train_classifier(dataset)

        elif type=='TRE':
            dataset = self.create_tre_data()
            self.model = TRE().to(self.device)
            self.w = self.train_TRE(dataset)

        elif type == 'random_uniform':
            self.w = torch.rand(*(self.x.shape[0],1)).cuda(self.device)

        elif type == 'ones':
            self.w = torch.ones(*(self.x.shape[0],1)).cuda(self.device)

    def retrain(self,x,z):
        self.x = x
        self.z = z
        if self.type == 'linear_classifier':
            dataset = self.create_classification_data()
            self.w = self.train_classifier(dataset)
        elif self.type == 'NCE':
            dataset = self.create_classification_data()
            self.w = self.train_classifier(dataset)

    def calc_loss(self,loss_func,pt,pf,target):
        if loss_func.__class__.__name__=='standard_bce':
            return loss_func(torch.cat([pt.squeeze(),pf.flatten()]),target)
        elif loss_func.__class__.__name__=='NCE_objective_stable':
            pf = pf.view(pt.shape[0], -1)
            return loss_func(pt,pf)


    def get_true_fake(self, dat_T, dat_F):

        pred_T = self.model(dat_T)
        pred_F = self.model(dat_F)
        return pred_T,pred_F


    def train_classifier(self,dataset):
        loss_func = NCE_objective_stable(self.kappa)
        # loss_func = standard_bce(pos_weight=self.kappa)
        opt = torch.optim.AdamW(self.model.parameters(),lr=self.est_params['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.5, patience=1)
        if self.est_params['mixed']:
            scaler = GradScaler()
        counter = 0
        best = np.inf
        idx = torch.randperm(dataset.X_val.shape[0])
        one_y = torch.ones(dataset.bs)
        zero_y = torch.zeros(dataset.bs*self.kappa)
        target = torch.cat([one_y,zero_y]).to(self.device)

        for i in range(self.est_params['max_its']):
            data_pos,data_neg= dataset.get_sample()
            opt.zero_grad()
            if self.est_params['mixed']:
                with autocast():
                    pt,pf = self.get_true_fake(data_pos,data_neg)
                    l = self.calc_loss(loss_func,pt,pf,target)
                scaler.scale(l).backward()
                scaler.step(opt)
                scaler.update()
            else:
                pt,pf = self.get_true_fake(data_pos,data_neg)
                l = self.calc_loss(loss_func, pt, pf, target)
                l.backward()
                opt.step()

            if i%(self.est_params['max_its']//25)==0:
                # print(l)
                with torch.no_grad():
                    dataset.val_mode()
                    self.model.eval()
                    one_y = torch.ones(dataset.bs)
                    zero_y = torch.zeros(dataset.bs)
                    target = torch.cat([one_y, zero_y]).to(self.device)
                    pt,pf = self.get_true_fake(torch.cat([dataset.X,dataset.Z],dim=1),torch.cat([dataset.X,dataset.Z[idx]],dim=1))
                    logloss = self.calc_loss(loss_func,pt,pf,target)
                    print(f'logloss epoch {i}: {logloss}')
                    scheduler.step(logloss)
                    if logloss.item()<best:
                        best = logloss.item()
                        counter=0
                        torch.save({'state_dict':self.model.state_dict(),
                                    'epoch':i},self.tmp_path+'best_run.pt')
                    else:
                        counter+=1
                    dataset.train_mode()
                    self.model.train()
                one_y = torch.ones(dataset.bs)
                zero_y = torch.zeros(dataset.bs * self.kappa)
                target = torch.cat([one_y, zero_y]).to(self.device)

            if counter>self.est_params['kill_counter']:
                print('stopped improving, stopping')
                break
        return self.model_eval()

    def model_eval(self):
        weights = torch.load(self.tmp_path+'best_run.pt')
        best_epoch = weights['epoch']
        print(f'loading best epoch {best_epoch}')
        self.model.load_state_dict(weights['state_dict'])
        self.model.eval()
        with torch.no_grad():
            w = self.model.get_w(torch.cat([self.x, self.z], dim=1))
            _w = w.cpu().squeeze().numpy()
            idx_HSIC = np.random.choice(np.arange(self.x.shape[0]), self.x.shape[0], p=_w / _w.sum())
            p_val = hsic_test(self.x[idx_HSIC, :], self.z[idx_HSIC, :], self.est_params['n_sample'])
            print(f'HSIC_pval : {p_val}')
            self.hsic_pval = p_val
            if p_val < self.est_params['criteria_limit']:
                self.failed = True
                print('failed')
        return w

    def forward_func_TRE(self,dat,indicator,y,loss_func):
        preds = self.model(dat,indicator)
        l = 0
        for p in preds:
            tmp = loss_func(p[~y,:],p[y,:])
            l+=tmp
        return l

    def train_TRE(self,dataset):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.est_params['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
        loss_func = NCE_objective_stable(kappa=1)
        if self.est_params['mixed']:
            scaler = GradScaler()
        counter = 0
        best = np.inf
        for i in range(self.est_params['max_its']):
            data,indicator,y= dataset.get_sample()
            opt.zero_grad()
            if self.est_params['mixed']:
                with autocast():
                    l = self.forward_func_TRE(data,indicator,y,loss_func)
                scaler.scale(l).backward()
                scaler.step(opt)
                scaler.update()
            else:
                l = self.forward_func_TRE(data, indicator, y, loss_func)
                l.backward()
                opt.step()

            if i%(self.est_params['max_its']//25)==0:
                # print(l)
                with torch.no_grad():
                    dataset.val_mode()
                    data, indicator, y = dataset.get_sample()
                    logloss = self.forward_func_TRE(data,indicator,y,loss_func)
                    print(f'logloss epoch {i}: {logloss}')
                    scheduler.step(logloss)
                    if logloss.item()<best:
                        best = logloss.item()
                        counter=0
                        torch.save({'state_dict': self.model.state_dict(),
                                    'epoch': i}, self.tmp_path + 'best_run.pt')
                    else:
                        counter+=1
                    dataset.train_mode()
            if counter>self.est_params['kill_counter']:
                print('stopped improving, stopping')
                break
        return self.model_eval()

    def create_tre_data(self):
        return classification_dataset_TRE(self.x,
                                          self.z,
                                          m=len(self.est_params['outputs']),
                                          p=1,
                                          bs=self.est_params['bs_ratio'],
                                          val_rate=self.est_params['val_rate'])

    def create_classification_data(self):
        self.kappa = self.est_params['kappa']
        return classification_dataset(self.x,
                                      self.z,
                                      bs=self.est_params['bs_ratio'],
                                      kappa=self.kappa,
                                      val_rate=self.est_params['val_rate'])

    def kernel_mean_matching(self):
        with torch.no_grad():
            list_idx = torch.from_numpy(get_i_not_j_indices(self.n)) #Seems to be working alright!
            list_idx=list_idx[torch.randperm(self.est_params['m']),:]
            torch_idx_x,torch_idx_z = list_idx.unbind(dim=1)
            data_extended = torch.cat([self.x[torch_idx_x],self.z[torch_idx_z]],dim=1)
            data = torch.cat([self.x,self.z],dim=1)
            self.kernel_ls_init('kernel_up', data)
            ls = self.get_median_ls_XY(data,data_extended)
            #self.n/list_idx.shape[0]*
            self.kernel_ls_init('kappa',ls=ls,data=data,data_2=data_extended)
            r3 = self.kappa.sum(1)*self.n/list_idx.shape[0]
            w,_= torch.solve(r3.unsqueeze(-1),self.kernel_up)
            return w

    def kernel_mean_matching_qp(self):
        with torch.no_grad():
            list_idx = torch.from_numpy(get_i_not_j_indices(self.n)) #Seems to be working alright!
            list_idx=list_idx[torch.randperm(self.est_params['m']),:]
            torch_idx_x,torch_idx_z = list_idx.unbind(dim=1)
            data_extended = torch.cat([self.x[torch_idx_x],self.z[torch_idx_z]],dim=1)
            data = torch.cat([self.x,self.z],dim=1)
            self.kernel_ls_init('kernel_up', data)
            ls = self.get_median_ls_XY(data,data_extended)
            #self.n/list_idx.shape[0]*
            self.kernel_ls_init('kappa',ls=ls,data=data,data_2=data_extended)
            r3 = self.kappa.sum(1).cpu().double().numpy()*self.n/list_idx.shape[0]
            self.kernel_up = self.kernel_up.cpu().double().numpy()
            w = kernel_mean_matching(self.kernel_up,r3,data_extended.shape[0])
            return torch.tensor(w).to(self.device)


    def return_weights(self):
        return self.w.squeeze()

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
            return ker()

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
            self.H = torch.eye(self.n)-torch.ones(self.n,1).repeat(1,self.n)/self.n
            if self.cuda:
                self.H = self.H.cuda(self.device)
            self.M = self.W*self.kernel_Y
            self.HMH = self.H@self.M@self.H

            # self.H_w = torch.diag(self.w)/self.n-self.w.repeat(1,self.n)/self.n**2
            # self.H_wXH_w = self.H_w@(self.kernel_X@self.H_w.t())
            # self.H_wYH_w = self.H_w@(self.kernel_Y@self.H_w.t())
            #
            # self.H_w1 = torch.diag(self.w)/self.n-torch.ones(*(self.n,self.n),device=self.device)/self.n**2
            # self.H_w1XH_w1 = self.H_w1@(self.kernel_X@self.H_w1.t())
            # self.H_w1YH_w1 = self.H_w1@(self.kernel_Y@self.H_w1.t())

    def get_permuted_HMH_kernel(self):
        idx = torch.randperm(self.n)
        kernel_Y = self.HMH[:,idx]
        kernel_Y = kernel_Y[idx,:]
        return kernel_Y,idx

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
            return self.n * torch.mean(self.kernel_X*self.HMH)
            # return (torch.sum(self.cache_W_K*self.kernel_Y)+self.middle_term-2*(self.w*self.cache_K_1 * self.cache_L_w).sum()/self.n**3)*self.n


    # def permute_Y_sanity(self):
    #     kernel_Y,_ = self.get_permuted_Y_kernel()
    #     return self.n*torch.sum(kernel_Y*self.H_wXH_w)
    #
    # def permute_Y_sanity_2(self):
    #     kernel_Y,_ = self.get_permuted_Y_kernel()
    #     return self.n*torch.sum(kernel_Y*self.H_w1XH_w1)


    def permute_X_sanity(self):
        # kernel_X,idx = self.get_permuted_X_kernel()
        HMH,_ = self.get_permuted_HMH_kernel()
        return self.n*torch.mean(HMH*self.kernel_X)
        # return self.n*torch.mean(self.HMH*kernel_X)

    #
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
            return self.permute_X_sanity()

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

class weighted_stat():
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






