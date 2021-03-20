import torch
import gpytorch
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
    def __init__(self, x, z,x_q, est_params=None, cuda=False, device=0, type='linear',secret_indx=0):
        self.x = x
        self.z = z
        self.cuda = cuda
        self.n = self.x.shape[0]
        self.device = device
        self.est_params = est_params
        self.type = type
        self.kernel_base = gpytorch.kernels.Kernel()
        self.tmp_path = f'./tmp_folder_{secret_indx}/'
        if os.path.exists(f'./tmp_folder_{secret_indx}/best_run.pt'):
            os.remove(f'./tmp_folder_{secret_indx}/best_run.pt')
        self.x_q = x_q
        if self.x_q.dim()==1:
            self.x_q = self.x_q.unsqueeze(-1)

        if not os.path.exists(self.tmp_path):
            os.makedirs(self.tmp_path)

        self.dataset = self.create_classification_data()
        self.dataset.set_mode('train')
        self.dataloader = NCE_dataloader(dataset=self.dataset,bs_ratio=self.est_params['bs_ratio'],shuffle=True,kappa=self.kappa,
                                    TRE=self.type in ['real_TRE','real_TRE_Q'])
        if self.type == 'NCE':
            self.model = MLP(d=self.dataset.X_train.shape[1]+self.dataset.Z_train.shape[1],f=self.est_params['width'],k=self.est_params['layers']).to(self.x.device)
            self.train_classifier()

        elif self.type=='NCE_Q':
            self.model = MLP(d=self.dataset.X_train.shape[1] + self.dataset.Z_train.shape[1], f=self.est_params['width'],k=self.est_params['layers']).to(self.x.device)
            self.train_classifier()

        elif self.type=='real_TRE':
            self.model = TRE_net(dim=self.dataset.X_train.shape[1]+self.dataset.Z_train.shape[1],
                                 o = 1,
                                 f=self.est_params['width'],
                                 k=self.est_params['layers'],
                                 m = self.est_params['m']
                                 ).to(self.x.device)
            self.train_classifier()
        elif self.type=='real_TRE_Q':
            self.model = TRE_net(dim=self.dataset.X_train.shape[1]+self.dataset.Z_train.shape[1],
                                 o = 1,
                                 f=self.est_params['width'],
                                 k=self.est_params['layers'],
                                 m = self.est_params['m']
                                 ).to(self.x.device)
            self.train_classifier()
        elif self.type=='rulsif':
            self.train_rulsif(self.dataset)
        # elif type == 'TRE_Q':
        #     self.model = MLP_pq(d_p=dataset.X_train.shape[1] + dataset.Z_train.shape[1],d_q=dataset.X_train.shape[1], f=self.est_params['width'], k=self.est_params['layers']).to(self.x.device)
        #     self.train_TRE_Q(dataset)
        elif self.type == 'random_uniform':
            self.w = torch.rand(*(self.x.shape[0],1)).squeeze().cuda(self.device)
        elif self.type == 'ones':
            self.w = torch.ones(*(self.x.shape[0],1)).squeeze().cuda(self.device)

    def create_classification_data(self):
        self.kappa = self.est_params['kappa']
        if self.type == 'NCE':
            return classification_dataset(X=self.x,
                                   Z=self.z,
                                    bs = self.est_params['bs_ratio'],
                                    kappa = self.kappa,
                                    val_rate = self.est_params['val_rate']
            )
        elif self.type=='NCE_Q':
            return classification_dataset_Q(X=self.x,
                                          Z=self.z,
                                          X_q=self.x_q,
                                          bs=self.est_params['bs_ratio'],
                                          kappa=self.kappa,
                                          val_rate=self.est_params['val_rate']
                                          )
        # elif self.type=='TRE_Q':
        #     return classification_dataset_Q_TRE(self.x,
        #                                     self.z,
        #                                     self.x_q,
        #                                     bs=self.est_params['bs_ratio'],
        #                                     kappa=self.kappa,
        #                                     val_rate=self.est_params['val_rate'],)
        elif self.type=='real_TRE':
            return dataset_MI_TRE(X=self.x,
                                            Z=self.z,
                                            m = self.est_params['m'],
                                            bs=self.est_params['bs_ratio'],
                                            val_rate=self.est_params['val_rate'],)
        elif self.type=='real_TRE_Q':
            return dataset_MI_TRE_Q(X=self.x,X_q=self.x_q,
                                            Z=self.z,
                                            m = self.est_params['m'],
                                            bs=self.est_params['bs_ratio'],
                                            val_rate=self.est_params['val_rate'],)

        elif self.type=='rulsif':
            return dataset_rulsif(X=self.x,X_q=self.x_q,Z=self.z,)

    def calc_loss(self,loss_func,pf,pt,target):
        if loss_func.__class__.__name__=='standard_bce':
            return loss_func(torch.cat([pt.squeeze(),pf.flatten()]),target)
        elif loss_func.__class__.__name__=='NCE_objective_stable':
            pf = pf.view(pt.shape[0], -1)
            return loss_func(pf,pt)

    def train_rulsif(self,dataset):
        pom,joint,pom_q = dataset.get_data()
        self.model = rulsif(pom=pom,joint=joint).to(self.device)
        self.model.calc_theta()



    def validation_loop(self,epoch):
        self.dataloader.dataset.set_mode('val')
        with torch.no_grad():
            pt = []
            pf = []
            for i,datalist in enumerate(self.dataloader):
                n = self.model.forward_val(datalist[0])
                p = self.model.forward_val(datalist[1])
                pf.append(n)
                pt.append(p)
            pt = torch.cat(pt,dim=0)
            pf = torch.cat(pf,dim=0)
            one_y = torch.ones_like(pt)
            zero_y = torch.zeros_like(pf)
            target = torch.cat([zero_y,one_y])
            logloss = self.calc_loss(self.loss_func,pf,pt,target)
            preds = torch.sigmoid(torch.cat([pf.squeeze(),pt.squeeze()]))
            auc = auc_check(  preds,target.squeeze())
            print(f'logloss epoch {epoch}: {logloss}')
            print(f'auc epoch {epoch}: {auc}')
            self.scheduler.step(logloss)
            if logloss.item()<self.best:
                self.best = logloss.item()
                self.counter=0
                torch.save({'state_dict':self.model.state_dict(),
                            'epoch':epoch},self.tmp_path+'best_run.pt')
            else:
                self.counter+=1
            if self.counter>self.est_params['kill_counter']:
                return True
            else:
                return False

    def train_loop(self):
        self.dataloader.dataset.set_mode('train')
        total_err = 0.
        for i,datalist in enumerate(self.dataloader):
            self.opt.zero_grad()
            l = self.forward_func(list_data=datalist, loss_func=self.loss_func)
            l.backward()
            self.opt.step()
            total_err+=l.item()
        print(f'train err: {total_err/i}')

    def forward_func(self, list_data, loss_func):
        list_preds = self.model(list_data)
        loss = 0
        for preds in list_preds:
            loss += self.calc_loss(loss_func, preds[0], preds[1], [])
        return loss / len(list_preds)

    def train_classifier(self):
        self.loss_func = NCE_objective_stable(self.kappa)
        self.opt = torch.optim.Adam(self.model.parameters(),lr=self.est_params['lr'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt,factor=0.5, patience=1)
        self.counter = 0
        self.best = np.inf
        torch.save({'state_dict': self.model.state_dict(),
                    'epoch': 0}, self.tmp_path + 'best_run.pt')
        for i in range(self.est_params['max_its']):
            print(f'epoch {i+1}')
            self.train_loop()
            if self.validation_loop(i+1):
                break

    def load_best_model(self):
        weights = torch.load(self.tmp_path + 'best_run.pt')
        best_epoch = weights['epoch']
        print(f'loading best epoch {best_epoch}')
        self.model.load_state_dict(weights['state_dict'])

    def model_eval(self,X,Z,X_q_test):
        if self.type in ['ones','random_uniform']:
            self.X_q_test = X_q_test
            if self.X_q_test.dim() == 1:
                self.X_q_test = self.X_q_test.unsqueeze(-1)
            return self.w
        n = X.shape[0]
        self.X_q_test = X_q_test
        if self.X_q_test.dim()==1:
            self.X_q_test = self.X_q_test.unsqueeze(-1)
        with torch.no_grad():
            if self.type == 'rulsif':
                w = self.model.get_w(X, Z, self.X_q_test)
            else:
                self.load_best_model()
                w = self.model.get_w(X, Z,[])
        _w = w.cpu().squeeze().numpy()
        return w

    def return_weights(self,X,Z,X_Q):
        self.w = self.model_eval(X,Z,X_Q)
        return self.w.squeeze()/self.kappa

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

class Q_weighted_HSIC(): # test-statistic seems to be to sensitive???
    def __init__(self,X,Y,w,X_q,cuda=False,device=0,half_mode=False,perm='Y',variant=1,seed=1):
        # torch.random.manual_seed(seed)
        self.X = X
        self.Y = Y
        self.perm = perm
        self.X_q = X_q
        self.n = X.shape[0]
        self.w = w.unsqueeze(-1) #works better with the bug haha
        self.cuda = cuda
        self.device = device
        self.half_mode = half_mode
        self.variant = variant
        if self.X_q.dim()==1:
            self.X_q = self.X_q.unsqueeze(-1)

        with torch.no_grad():
            self.kernel_X_X_q = gpytorch.kernels.RBFKernel().cuda(device)
            _ls = self.get_median_ls(self.X,self.X_q) #This is critical, figure out way to get the right lr
            self.kernel_X_X_q._set_lengthscale(_ls)
            self.k_X_X_q = self.kernel_X_X_q(self.X, self.X_q).evaluate()
            for name, data in zip(['X', 'Y','X_q'], [self.X, self.Y,self.X_q]):
                self.kernel_ls_init(name, data)

            self.const_var_1_X = self.kernel_Y @ self.w
            self.const_var_1_Y = self.k_X_X_q.sum(dim=1, keepdim=True)
            self.const_sum_Y = self.kernel_X_q.sum()
            self.a_2 = self.const_sum_Y * (self.w.t() @ self.const_var_1_X) / self.n ** 4

    def get_permuted2d(self,ker):
        idx = torch.randperm(self.n)
        kernel_X = ker[:,idx]
        kernel_X = kernel_X[idx,:]
        return kernel_X,idx

    def get_permuted1d(self,ker):
        idx = torch.randperm(self.n)
        kernel_X = ker[idx,:]
        return kernel_X,idx

    def calculate_weighted_statistic(self):
        with torch.no_grad():
            a_1 = self.w.t() @ (self.kernel_X * self.kernel_Y) @ self.w / self.n ** 2
            a_3 = self.w.t() @ (self.const_var_1_Y * (self.const_var_1_X)) / self.n ** 3
            return self.n*(a_1+self.a_2-2*a_3).squeeze()

    def permutation_calculate_weighted_statistic(self):
        with torch.no_grad():
            if self.perm =='X':
                kernel_X,idx = self.get_permuted2d(self.kernel_X)
                a_1 = self.w.t() @ (kernel_X * self.kernel_Y) @ self.w / self.n ** 2
                a_3 = self.w.t() @ (self.const_var_1_Y[idx, :] * self.const_var_1_X) / self.n ** 3
                return self.n * (a_1 + self.a_2 - 2 * a_3)
            else:
                kernel_Y, idx = self.get_permuted2d(self.kernel_Y)
                a_1 = self.w.t() @ (self.kernel_X * kernel_Y) @ self.w / self.n ** 2
                a_3 = self.w.t() @ (self.const_var_1_Y * self.const_var_1_X[idx, :]) / self.n ** 3
                a_2 = self.const_sum_Y * (self.w.t() @ self.const_var_1_X[idx, :]) / self.n ** 4
                return self.n * (a_1 + a_2 - 2 * a_3)

    def kernel_ls_init(self, name, data):
        setattr(self, f'ker_obj_{name}',
                gpytorch.kernels.RBFKernel().cuda(self.device) if self.cuda else gpytorch.kernels.RBFKernel())
        ls = self.get_median_ls(data)
        getattr(self, f'ker_obj_{name}')._set_lengthscale(ls)
        setattr(self, f'kernel_{name}', getattr(self, f'ker_obj_{name}')(data).evaluate())

    def get_median_ls(self, X,Y=None):
        kernel_base = gpytorch.kernels.Kernel().cuda(self.device)
        with torch.no_grad():
            if Y is None:
                d = kernel_base.covar_dist(x1=X, x2=X)
            else:
                d = kernel_base.covar_dist(x1=X, x2=Y)
            ret = torch.sqrt(torch.median(d[d > 0])) # print this value, should be increasing with d
            return ret

class weighted_HSIC():
    def __init__(self,X,Y,w,device=0,perm='Y',variant =1):
        torch.cuda.set_device(device)
        self.cuda= True
        self.device = device
        self.X = X
        self.Y = Y
        self.perm = perm
        self.w = w.unsqueeze(-1)
        self.variant  = variant
        self.n = self.X.shape[0]
        self.H_w = torch.diag(self.w.squeeze())-1/self.n *self.w.expand(-1,self.n)
        self.W = self.w@self.w.t()
        self.H = torch.eye(self.n).cuda(device) - 1/self.n * torch.ones_like(self.W)

        with torch.no_grad():
            for name, data in zip(['X', 'Y'], [self.X, self.Y]):
                self.kernel_ls_init(name, data)
            if self.variant == 1:
                self.const_var_1_X = self.kernel_Y@self.w
                self.const_var_1_Y = self.kernel_X.sum(dim=1,keepdim=True)
                self.const_sum_Y = self.kernel_X.sum()
                self.a_2 = self.const_sum_Y*(self.w.t()@self.const_var_1_X)/self.n**4

            elif self.variant == 2:
                self.const_var_2 = self.H_w@self.kernel_X@self.H_w.t()
            elif self.variant == 3:
                self.const_var_3 = self.H_w.t() @ self.kernel_Y @ self.H_w
            elif self.variant == 4:
                self.M = self.W * self.kernel_Y
                self.const_var_4_X = self.H@self.M@self.H

    def get_permuted2d(self,ker):
        idx = torch.randperm(self.n)
        kernel_X = ker[:,idx]
        kernel_X = kernel_X[idx,:]
        return kernel_X,idx

    def get_permuted1d(self,ker):
        idx = torch.randperm(self.n)
        kernel_X = ker[idx,:]
        return kernel_X,idx

    def calculate_weighted_statistic(self):
        with torch.no_grad():
            if self.variant ==1:
                a_1  =  self.w.t()@(self.kernel_X*self.kernel_Y)@self.w/self.n**2
                a_3 = self.w.t()@(self.const_var_1_Y*(self.const_var_1_X))/self.n**3
                return self.n*(a_1+self.a_2-2*a_3)
            elif self.variant == 2:
                return 1/self.n * torch.sum(self.const_var_2 * self.kernel_Y)
            elif self.variant ==3:
                return 1 / self.n * torch.sum(self.const_var_3 * self.kernel_X)
            elif self.variant ==4:
                return 1 / self.n * torch.sum(self.kernel_X *self.const_var_4_X)

    def permutation_calculate_weighted_statistic(self):
        with torch.no_grad():
            if self.perm=='X':
                kernel_X,idx = self.get_permuted2d(self.kernel_X)
                if self.variant == 1:
                    a_1 = self.w.t() @ (kernel_X * self.kernel_Y) @ self.w / self.n ** 2
                    a_3 = self.w.t() @ (self.const_var_1_Y[idx,:] * self.const_var_1_X) / self.n ** 3
                    return self.n * (a_1 + self.a_2 - 2 * a_3)
                elif self.variant == 2:
                    return 1 / self.n * torch.sum(self.H_w @ kernel_X @ self.H_w.t() * self.kernel_Y)
                elif self.variant == 3:
                    return 1 / self.n * torch.sum(self.const_var_3  * kernel_X)
                elif self.variant == 4:
                    return 1 / self.n * torch.sum(kernel_X * self.const_var_4_X)
            else:
                if self.variant == 1:
                    kernel_Y, idx = self.get_permuted2d(self.kernel_Y)
                    a_1 = self.w.t() @ (self.kernel_X * kernel_Y) @ self.w / self.n ** 2
                    a_3 = self.w.t() @ (self.const_var_1_Y * self.const_var_1_X[idx,:]) / self.n ** 3
                    a_2 = self.const_sum_Y*(self.w.t()@self.const_var_1_X[idx,:])/self.n**4
                    return self.n * (a_1 + a_2 - 2 * a_3)
                elif self.variant == 2:
                    kernel_Y, idx = self.get_permuted2d(self.kernel_Y)
                    return 1 / self.n * torch.sum(self.const_var_2 * kernel_Y)
                elif self.variant == 3:
                    kernel_Y, idx = self.get_permuted2d(self.kernel_Y)
                    return 1 / self.n * torch.sum( self.H_w.t() @ kernel_Y @ self.H_w* self.kernel_X)
                elif self.variant == 4:
                    M, idx = self.get_permuted2d(self.M)
                    return 1 / self.n * torch.sum(self.kernel_X * self.H@M@self.H)


    def kernel_ls_init(self, name, data):
        setattr(self, f'ker_obj_{name}',
                gpytorch.kernels.RBFKernel().cuda(self.device) if self.cuda else gpytorch.kernels.RBFKernel())
        ls = self.get_median_ls(data)
        getattr(self, f'ker_obj_{name}')._set_lengthscale(ls)
        setattr(self, f'kernel_{name}', getattr(self, f'ker_obj_{name}')(data).evaluate())

    def get_median_ls(self, X,Y=None):
        kernel_base = gpytorch.kernels.Kernel().cuda(self.device)
        with torch.no_grad():
            if Y is None:
                d = kernel_base.covar_dist(x1=X, x2=X)
            else:
                d = kernel_base.covar_dist(x1=X, x2=Y)
            ret = torch.sqrt(torch.median(d[d > 0])) # print this value, should be increasing with d
            return ret



class weighted_stat():
    def __init__(self,X,Y,w,do_null = True,cuda=False,device=0,half_mode=False):
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
    def __init__(self,X,Y,w,do_null = True,cuda=False,device=0):
        super(weighted_statistic_new, self).__init__(X, Y, w, do_null, cuda, device)
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






