import torch
from kgformula.utils import calculate_power,hypothesis_acceptance
import gpytorch
from torch.distributions import Normal,StudentT,Bernoulli,Beta,Uniform,Exponential
import copy
class weighted_stat():
    def __init__(self,X,Y,Z,w,do_null = True,reg_lambda=1e-3,cuda=False,device=0):
        self.n = X.shape[0]
        self.H = torch.ones(*(self.n, self.n)) * (1 - 1 / self.n)
        if cuda:
            self.H = self.H.cuda(device)
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
        for name,data in zip(['kernel_X','kernel_Y'],[X,Y]):
            self.kernel_ls_init(name,data)

    def kernel_ls_init(self,name,data):
        ker = gpytorch.kernels.RBFKernel()
        ls = self.get_median_ls(data)
        ker._set_lengthscale(ls)
        if self.cuda:
            ker = ker.cuda(self.device)
        setattr(self,name,ker)

    def get_median_ls(self,X):
        with torch.no_grad():
            d = self.kernel_base.covar_dist(x1=X,x2=X)
            ret = torch.sqrt(torch.median(d[d > 0]))
            return ret
    def calculate_statistic(self):
        with torch.no_grad():
            X_ker = self.kernel_X(self.X)
            Y_ker = self.kernel_Y(self.Y)
            return (X_ker@self.H@Y_ker).sum()

    def permutation_calculate_weighted_statistic(self):
        idx = torch.randperm(self.n)
        with torch.no_grad():
            X_ker = self.kernel_X(self.X) * self.w
            Y_ker = self.kernel_Y(self.Y)[idx]
            centered = X_ker@self.H
            return (centered@Y_ker.evaluate()).sum()

    def calculate_weighted_statistic(self):
        with torch.no_grad():
            X_ker = self.kernel_X(self.X) * self.w
            Y_ker = self.kernel_Y(self.Y)
            centered = X_ker@self.H
            return (centered@Y_ker.evaluate()).sum()

class weigted_statistic_new(weighted_stat):
    def __init__(self,X,Y,Z,w,do_null = True,reg_lambda=1e-3,cuda=False,device=0):
        super(weigted_statistic_new, self).__init__(X, Y, Z, w, do_null, reg_lambda, cuda, device)

    def permutation_calculate_weighted_statistic(self):
        idx = torch.randperm(self.n)
        with torch.no_grad():
            X_ker = self.kernel_X(self.X)
            Y_ker = self.kernel_Y(self.Y)[idx]
            A = self.H@gpytorch.matmul(X_ker,self.H)*(self.H@gpytorch.matmul(Y_ker,self.H))
            return A.sum()

    def calculate_weighted_statistic(self):
        with torch.no_grad():
            X_ker = self.kernel_X(self.X)
            Y_ker = self.kernel_Y(self.Y)
            A = self.H@gpytorch.matmul(X_ker,self.H)*(self.H@gpytorch.matmul(Y_ker,self.H))
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
        for name,data in zip(['kernel_X','kernel_Y','kernel_Z'],[X,Y,Z]):
            self.kernel_ls_init(name,data)

        if distribution=='normal':
            self.d = Normal(0,1)
        elif distribution=='uniform':
            self.d = Uniform(0,1)
        elif distribution=='exp':
            self.d = Exponential(1)
        elif distribution=='ber':
            self.d = Bernoulli(0.5)

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
        setattr(self,name,ker)

    def get_median_ls(self,X):
        with torch.no_grad():
            d = self.kernel_base.covar_dist(x1=X,x2=X)
            ret = torch.sqrt(torch.median(d[d > 0]))
            return ret

    def calculate_base_components(self):
        with torch.no_grad():
            M = self.kernel_Z(self.Z).evaluate()
            Mav = M@self.ones@self.ones.t()
            W = torch.solve(self.kernel_X(self.X)*M+self.diag,Mav)
            self.C = self.cov(W)

    def calculate_weighted_statistic(self):
        return torch.sum(self.C*self.kernel_Y(self.Y))

    def permutation_calculate_weighted_statistic(self):
        W = self.sample_W()
        return torch.sum(W*self.C*self.kernel_Y(self.Y))

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




