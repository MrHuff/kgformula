import torch
import gpytorch
from torch.distributions import Normal,StudentT,Bernoulli,Beta,Uniform,Exponential
import numpy as np
from pykeops.torch import LazyTensor,Genred
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn import metrics
import torch.nn as nn
def auc_check(y_pred,Y):
    with torch.no_grad():
        y_pred = (y_pred.float() > 0.5).cpu().float().numpy()
        fpr, tpr, thresholds = metrics.roc_curve(Y.cpu().numpy(), y_pred, pos_label=1)
        auc =  metrics.auc(fpr, tpr)
        return auc

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
    def __init__(self,d,f=12,k=2):
        super(MLP, self).__init__()
        self.model = nn.ModuleList()
        self.model.append(nn.Linear(d, d*f))
        self.model.append(nn.Tanh())
        for i in range(k):
            self.model.append(nn.Linear(d*f, d*f))
            self.model.append(nn.Tanh())
        self.model.append(nn.Linear(d*f, 1))
        self.model.append(nn.Tanh())

    def forward(self,x):
        for l in self.model:
            x = l(x)
        return x

    def logistic_forward(self,x):
        return torch.nn.functional.sigmoid(self.W(x))

class logistic_regression(torch.nn.Module):
    def __init__(self,d):
        super(logistic_regression, self).__init__()
        self.W = torch.nn.Linear(in_features=d,  out_features=1,bias=True)

    def forward(self,x):
        return self.W(x)

    def logistic_forward(self,x):
        return torch.nn.functional.sigmoid(self.W(x))

class classification_dataset(Dataset):
    def __init__(self,X,y):
        super(classification_dataset, self).__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i,:],self.y[i]

class density_estimator():
    def __init__(self, x, z, est_params=None, reg_lambda=1e-3, cuda=False, device=0, type='linear'):
        self.failed = False
        self.x = x
        self.z = z
        self.cuda = cuda
        self.n = self.x.shape[0]
        self.device = device
        self.diag = reg_lambda*torch.eye(self.n)
        self.est_params = est_params
        if self.cuda:
            self.diag = self.diag.cuda(self.device)
        self.kernel_base = gpytorch.kernels.Kernel()

        if type=='linear':
            self.alpha = est_params['alpha']
            self.linear_x_of_z()
            self.get_w_kdre()

        elif type=='gp':
            self.alpha = est_params['alpha']
            self.kernel_ls_init('kernel_tmp', self.z)
            self.gp_x_of_z()
            self.get_w_kdre()

        elif type=='semi':
            self.semi_cheat_x_of_z()

        elif type == 'kmm':
            self.kernel_mean_matching()

        elif type == 'classifier':
            dataset = self.create_classification_data()
            self.model = MLP(d=dataset.X.shape[1],f=256,k=1).to(self.x.device)
            self.w = self.train_classifier(dataset)

    def train_classifier(self,dataset):
        # dataloader = DataLoader(dataset,batch_size=self.est_params['batch_size'],shuffle=True)
        loss_func = torch.nn.BCEWithLogitsLoss()
        opt = torch.optim.Adam(self.model.parameters(),lr=self.est_params['lr'])
        # opt = torch.optim.LBFGS(self.model.parameters(),lr=1e-1)
        auc=0
        # for j in range(self.est_params['epochs']):
        j = 0
        while auc<0.9:

            X = dataset.X
            y = dataset.y
        # for i,(X,y) in enumerate(dataloader):
            opt.zero_grad()
            pred = self.model(X)
            l = loss_func(pred.squeeze(),y.squeeze())
            l.backward()
            opt.step()
                # def closure():
                #     pred = self.model(X)
                #     opt.zero_grad()
                #     l.backward()
                #     l = loss_func(pred.squeeze(),y)
                #     return l
                # opt.step(closure)
            if j%50==0:
                with torch.no_grad():
                    pred = self.model(X)
                    auc = auc_check(pred.squeeze(),y.squeeze())
                    # print(f'auc epoch {j}: {auc}')
            j+=1
            if j>self.est_params['max_its']:
                self.failed = True
                print('failed')
                break
        with torch.no_grad():
            p = self.model(self.data_pos)
        return (1-p)/p

    def create_classification_data(self):
        with torch.no_grad():
            # list_idx = torch.from_numpy(get_i_not_j_indices(self.n))  # Seems to be working alright!
            # torch_idx_x, torch_idx_z = list_idx.unbind(dim=1)
            perm = torch.randperm(self.x.shape[0])
            data_neg = torch.cat([self.x, self.z[perm]], dim=1)
            # idx = np.random.choice(data_neg.shape[0],self.est_params['sigma_n'])
            # data_neg = data_neg[idx,:]
            neg_samples = torch.zeros(data_neg.shape[0]).to(self.x.device)
            pos_samples = torch.ones(self.x.shape[0]).to(self.x.device)
            self.data_pos = torch.cat([self.x, self.z], dim=1)
            X = torch.cat([self.data_pos,data_neg],dim=0)
            y = torch.cat([pos_samples,neg_samples],dim=0)
        return classification_dataset(X,y)

    def kernel_mean_matching(self):
        with torch.no_grad():
            list_idx = torch.from_numpy(get_i_not_j_indices(self.n)) #Seems to be working alright!
            torch_idx_x,torch_idx_z = list_idx.unbind(dim=1)
            data_extended = torch.cat([self.x[torch_idx_x],self.z[torch_idx_z]],dim=1)
            data = torch.cat([self.x,self.z],dim=1)
            self.kernel_ls_init('kernel_up', data)
            ls = self.get_median_ls_XY(data,data_extended)
            # self.kernel_ls_init('kernel_down',data,data_extended,ls)
            # y = self.kernel_down.sum(dim=1).unsqueeze(-1)/(self.n-1)
            y = self.kernel_ls_init_keops(ls=ls,data=data,data_2=data_extended)
            self.w,_ = torch.solve(y,(self.kernel_up+self.diag))

    def get_w_kdre(self):
        self.kernel_ls_init('kernel_up', self.x)
        self.kernel_ls_init('kernel_down', self.x, self.down_estimator)
        with torch.no_grad():
            self.h_hat = self.kernel_up.mean(dim=1,keepdim=True)
            self.H = self.alpha/self.n * torch.mm(self.kernel_up, self.kernel_up) + (1-self.alpha)/self.n * torch.mm(self.kernel_down, self.kernel_down) + self.diag
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

class weighted_stat(): #HAPPY MISTAKE?!?!??!?!?!?!?!?
    def __init__(self,X,Y,Z,w,do_null = True,reg_lambda=1e-3,cuda=False,device=0,half_mode=False):
        with torch.no_grad():
            self.device = device
            self.cuda = cuda
            self.n = X.shape[0]
            self.H = torch.ones(*(self.n, 1)) * (1 - 1 / self.n)
            self.H_2 = torch.eye(self.n)-2*torch.ones(*(self.n, self.n))/self.n
            self.H_4 = torch.eye(self.n)-4*torch.ones(*(self.n, self.n))/self.n
            self.one_n_1 = torch.ones(*(self.n,1))
            self.ones= torch.ones(*(self.n,self.n))
            self.half_mode = half_mode
            if cuda:
                self.H = self.H.cuda(device)
                self.H_2 = self.H_2.cuda(device)
                self.H_4 = self.H_4.cuda(device)
                self.ones = self.ones.cuda(device)
                self.one_n_1 = self.one_n_1.cuda(device)
            else:
                self.device = 'cpu'
            self.X = X
            self.Y = Y
            self.Z = Z
            self.w = w.unsqueeze(-1)
            self.W = self.w@self.w.t()
            self.W = self.W/self.n
            self.do_null=do_null
            self.kernel_base = gpytorch.kernels.Kernel()
            self.reg_lambda = reg_lambda
            for name,data in zip(['X','Y'],[self.X,self.Y]):
                self.kernel_ls_init(name,data)
            if self.half_mode:
                self.H = self.H.half()
                self.w = self.w.half()
                self.W = self.W.half()
            self.X_ker = self.kernel_X
            self.Y_ker = self.kernel_Y
            self.center_X = (self.X_ker@self.H)*self.w
            self.center_Y = self.kernel_Y@self.H

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

    def permutation_calculate_weighted_statistic(self):
        idx = torch.randperm(self.n)
        with torch.no_grad():
            return (self.center_X*self.center_Y[idx]).sum() #WHY WOULD THIS MATTER?!?!?!? [idx] placement is really weird

    def calculate_weighted_statistic(self):
        with torch.no_grad():
            return (self.center_X*self.center_Y).sum()

class weighted_statistic_new(weighted_stat):
    def __init__(self,X,Y,Z,w,do_null = True,reg_lambda=1e-3,cuda=False,device=0):
        super(weighted_statistic_new, self).__init__(X, Y, Z, w, do_null, reg_lambda, cuda, device)
        with torch.no_grad():
            self.sum_mean_X = self.X_ker.mean()
            self.X_ker_H_4 = self.X_ker@self.H_4
            self.X_ker_H_2= self.X_ker@self.H_2
            self.X_ker_n_1 = self.X_ker@self.one_n_1
            self.X_ker_n_1 = self.X_ker_n_1/self.n
            self.X_ker_ones =self.X_ker@self.ones
            self.X_ker_ones =self.X_ker_ones/self.n

            self.sum_mean_Y = self.Y_ker.mean()
            self.Y_ker_H_4 = self.Y_ker@self.H_4
            self.Y_ker_H_2= self.Y_ker@self.H_2
            self.Y_ker_n_1 = self.Y_ker@self.one_n_1
            self.Y_ker_ones =self.Y_ker@self.ones
            self.Y_ker_n_1 = self.Y_ker_n_1/self.n
            self.Y_ker_ones =self.Y_ker_ones/self.n

            self.term_1 = 0.5*self.W*self.X_ker_H_4
            self.term_2 = 0.5*self.W*self.X
            self.term_3 = 2*self.W*self.X_ker_ones
            self.term_4 = 2*self.W
            self.term_5 = self.W*self.sum_mean_X
            self.term_6 = self.W*self.sum_mean_Y*self.X_ker_H_2
            self.term_7 = self.W*self.sum_mean_Y*self.sum_mean_X

    def permutation_calculate_weighted_statistic(self):
        with torch.no_grad():
            idx = torch.randperm(self.n)
            y = self.Y[idx]
            Y_ker = self.ker_obj_Y(y).evaluate()
            Y_ker_H_4 = Y_ker @ self.H_4
            Y_ker_H_2 = Y_ker @ self.H_2
            Y_ker_n_1 = Y_ker @ self.one_n_1
            Y_ker_n_1 = Y_ker_n_1/self.n
            Y_ker_ones = Y_ker @ self.ones
            Y_ker_ones = Y_ker_ones/self.n
            test_stat = self.term_1 * Y_ker + self.term_2 * Y_ker_H_4 + self.term_3 * Y_ker_ones + self.term_4 * (
                        self.X_ker_n_1 @ Y_ker_n_1.t()) + self.term_5 * Y_ker_H_2 + self.term_6 + self.term_7
            return test_stat.sum()

    def calculate_weighted_statistic(self):
        with torch.no_grad():
            test_stat = self.term_1 * self.Y_ker + self.term_2 * self.Y_ker_H_4 + self.term_3 * self.Y_ker_ones + self.term_4 * (
                    self.X_ker_n_1 @ self.Y_ker_n_1.t()) + self.term_5 * self.Y_ker_H_2 + self.term_6 + self.term_7
            return test_stat.sum()

class wild_bootstrap_deviance():
    def __init__(self,X,Y,Z,distribution='normal',do_null = True,reg_lambda=1e-3,cuda=False,device=0):
        self.n = X.shape[0]
        self.reg_lambda = reg_lambda
        self.diag = self.reg_lambda * torch.eye(self.n)
        self.H = torch.ones(*(self.n, self.n)) * (1 - 1 / self.n)
        self.ones = torch.ones(*(self.n,1))
        if cuda:
            self.H = self.H.cuda(device)
            self.diag = self.diag.cuda(device)
            self.ones = self.ones.cuda(device)
        else:
            device = 'cpu'
        self.X = X if not cuda else X.cuda(device)
        self.Y = Y if not cuda else Y.cuda(device)
        self.Z = Z if not cuda else Z.cuda(device)
        self.device = device
        self.cuda = cuda
        self.do_null=do_null
        self.kernel_base = gpytorch.kernels.Kernel()
        for name,data in zip(['kernel_X','kernel_Y','kernel_Z'],[self.X,self.Y,self.Z]):
            self.kernel_ls_init(name,data)

        if distribution=='normal':
            self.d = Normal(0,1)
        elif distribution=='uniform':
            self.d = Uniform(0,1)
        elif distribution=='exp':
            self.d = Exponential(1)
        elif distribution=='ber':
            self.d = Bernoulli(0.5)
        self.calculate_base_components()

    def sample_W(self):
        w = self.d.sample((self.n,1))
        W = w@w.t()
        if self.cuda:
            W = W.cuda(self.device)
        return W

    def kernel_ls_init(self,name,data):
        ker = gpytorch.kernels.RBFKernel()
        ls = self.get_median_ls(data)
        ker._set_lengthscale(ls)
        if self.cuda:
            ker = ker.cuda(self.device)
        setattr(self,name,ker(data).evaluate())

    def get_median_ls(self,X):
        with torch.no_grad():
            d = self.kernel_base.covar_dist(x1=X,x2=X)
            ret = torch.sqrt(torch.median(d[d > 0]))
            return ret

    def calculate_base_components(self):
        with torch.no_grad():
            Mav = self.kernel_Z@self.ones@self.ones.t()
            W,_ = torch.solve(Mav,self.kernel_X*self.kernel_Z+self.diag)
            self.C = self.cov(W)

    def calculate_weighted_statistic(self):
        return torch.sum(self.C*self.kernel_Y)

    def permutation_calculate_weighted_statistic(self):
        W = self.sample_W()
        return torch.sum(W*self.C*self.kernel_Y)

    def cov(self,m, rowvar=False):
        '''Estimate a covariance matrix given data.

        Covariance indicates the level to which two variables vary together.
        If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
        then the covariance matrix element `C_{ij}` is the covariance of
        `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

        Args:
            m: A 1-D or 2-D array containing multiple variables and observations.
                Each row of `m` represents a variable, and each column a single
                observation of all those variables.
            rowvar: If `rowvar` is True, then each row represents a
                variable, with observations in the columns. Otherwise, the
                relationship is transposed: each column represents a variable,
                while the rows contain observations.

        Returns:
            The covariance matrix of the variables.
        '''
        if m.dim() > 2:
            raise ValueError('m has more than 2 dimensions')
        if m.dim() < 2:
            m = m.view(1, -1)
        if not rowvar and m.size(0) != 1:
            m = m.t()
        # m = m.type(torch.double)  # uncomment this line if desired
        fact = 1.0 / (m.size(1) - 1)
        m -= torch.mean(m, dim=1, keepdim=True)
        mt = m.t()  # if complex: mt = m.t().conj()
        return fact * m.matmul(mt).squeeze()




