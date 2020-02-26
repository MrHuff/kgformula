import torch
from kgformula.utils import calculate_pval,hypothesis_acceptance
import gpytorch
from torch.distributions import Normal,StudentT,Bernoulli,Beta,Uniform,Exponential
import copy


class density_estimator():
    def __init__(self,numerator_sample,denominator_sample,alpha=0.5,reg_lambda=1e-3,cuda=False,device=0,type='linear'):
        self.up = numerator_sample if not cuda else numerator_sample.cuda(device)
        self.down = denominator_sample if not cuda else denominator_sample.cuda(device)
        self.cuda = cuda
        self.n = self.up.shape[0]
        self.device = device
        self.diag = reg_lambda*torch.eye(self.n)
        if self.cuda:
            self.diag = self.diag.cuda()
        self.kernel_base = gpytorch.kernels.Kernel()

        if type=='linear':
            self.linear_x_of_z()
        elif type=='gp':
            self.kernel_ls_init('kernel_tmp', self.down)
            self.gp_x_of_z()
        self.kernel_ls_init('kernel_up',self.up)
        self.kernel_ls_init('kernel_down',self.up,self.down_estimator)

        with torch.no_grad():
            self.h_hat = self.kernel_up.mean(dim=1,keepdim=True)
            self.H = alpha/self.n * torch.mm(self.kernel_up, self.kernel_up) + (1-alpha)/self.n * torch.mm(self.kernel_down, self.kernel_down) + self.diag
            self.w,_ = torch.solve(self.h_hat, self.H)

    def return_weights(self):
        return self.w.squeeze()

    def linear_x_of_z(self):
        self.down = torch.cat([self.down,torch.ones_like(self.down)],dim=1)
        with torch.no_grad():
            self.down_estimator = self.down@(torch.inverse(self.down.t()@self.down)@(self.down.t()@self.up))

    def gp_x_of_z(self):
        with torch.no_grad():
            s,_ = torch.solve(self.up,self.kernel_tmp+self.diag)
            self.down_estimator = self.kernel_tmp@s

    def get_median_ls(self,X):
        with torch.no_grad():
            d = self.kernel_base.covar_dist(x1=X,x2=X)
            ret = torch.sqrt(torch.median(d[d > 0]))
            return ret

    def kernel_ls_init(self,name,data,data_2=None):
        ker = gpytorch.kernels.RBFKernel()
        ls = self.get_median_ls(data)
        ker._set_lengthscale(ls)
        if self.cuda:
            ker = ker.cuda(self.device)
            if data_2 is None:
                setattr(self,name,ker(data).evaluate())
            else:
                setattr(self,name,ker(data,data_2).evaluate())

class weighted_stat(): #HAPPY MISTAKE?!?!??!?!?!?!?!?
    def __init__(self,X,Y,Z,w,do_null = True,reg_lambda=1e-3,cuda=False,device=0,half_mode=False):
        with torch.no_grad():
            self.n = X.shape[0]
            self.H = torch.ones(*(self.n, 1)) * (1 - 1 / self.n)
            self.H_2 = torch.eye(self.n)-torch.ones(*(self.n, self.n))/self.n

            self.half_mode = half_mode
            if cuda:
                self.H = self.H.cuda(device)
                self.H_2 = self.H_2.cuda(device)

            else:
                device = 'cpu'
            self.X = X if not cuda else X.cuda(device)
            self.Y = Y if not cuda else Y.cuda(device)
            self.Z = Z if not cuda else Z.cuda(device)
            self.w = w.unsqueeze(-1) if not cuda else w.unsqueeze(-1).cuda(device)
            self.W = self.w@self.w.t()
            self.device = device
            self.cuda = cuda
            self.do_null=do_null
            self.kernel_base = gpytorch.kernels.Kernel()
            self.reg_lambda = reg_lambda
            for name,data in zip(['kernel_X','kernel_Y'],[self.X,self.Y]):
                self.kernel_ls_init(name,data)
            if self.half_mode:
                self.H = self.H.half()
                self.w = self.w.half()
                self.W = self.W.half()
            self.X_ker = self.kernel_X*self.w
            self.center_X = self.X_ker@self.H
            self.center_Y = self.kernel_Y@self.H

    def kernel_ls_init(self,name,data):
        ker = gpytorch.kernels.RBFKernel()
        ls = self.get_median_ls(data)
        ker._set_lengthscale(ls)
        if self.cuda:
            ker = ker.cuda(self.device)
        if self.half_mode:
            setattr(self,name,ker(data).evaluate().half())
        else:
            setattr(self,name,ker(data).evaluate())

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

class weigted_statistic_new(weighted_stat):
    def __init__(self,X,Y,Z,w,do_null = True,reg_lambda=1e-3,cuda=False,device=0):
        super(weigted_statistic_new, self).__init__(X, Y, Z, w, do_null, reg_lambda, cuda, device)
        with torch.no_grad():
            self.center_X = self.H_2@(self.kernel_X@self.H_2)
            self.X_W = self.center_X@self.W
            self.center_Y = self.H_2@(self.kernel_Y@self.H_2)

    def permutation_calculate_weighted_statistic(self):
        idx = torch.randperm(self.n)
        with torch.no_grad():
            A = self.X_W*self.center_Y[idx]
            return A.sum()

    def calculate_weighted_statistic(self):
        with torch.no_grad():
            A = self.X_W*self.center_Y
            return A.sum()

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




