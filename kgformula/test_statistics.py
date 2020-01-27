import torch
from kgformula.utils import calculate_power,hypothesis_acceptance
import gpytorch
from torch.distributions.studentT import StudentT
import copy
class weighted_stat():
    def __init__(self,X,Y,Z,do_null = True,get_p_x_cond_z=None,reg_lambda=1e-3):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.n = X.shape[0]
        self.do_null=do_null
        self.get_p_x_cond_z = get_p_x_cond_z
        self.kernel_base = gpytorch.kernels.Kernel()
        self.reg_lambda = reg_lambda
        self.H = torch.ones(*(self.n,self.n))*(1-1/self.n)
        for name,data in zip(['kernel_X','kernel_Y','kernel_Z'],[X,Y,Z]):
            self.kernel_ls_init(name,data)
        self.get_weights()

    def kernel_ls_init(self,name,data):
        ker = gpytorch.kernels.RBFKernel()
        ls = self.get_median_ls(data)
        ker._set_lengthscale(ls)
        setattr(self,name,ker)

    def get_median_ls(self,X):
        with torch.no_grad():
            d = self.kernel_base.covar_dist(x1=X,x2=X)
            return torch.sqrt(torch.median(d[d > 0]))

    def cme_x_given_z(self):
        Z_ker = self.kernel_Z(self.Z)
        X_ker = self.kernel_X(self.X)
        with torch.no_grad():
            mat = gpytorch.add_diag(Z_ker, self.n * self.reg_lambda)
            mu_embed = gpytorch.inv_matmul(mat,Z_ker,left_tensor=X_ker) #nxn matrix - grid of the mean embedding of X|Z
            return mu_embed

    def get_weights(self):
        with torch.no_grad():
            if self.get_p_x_cond_z is not None:
                self.w = 1 / self.get_p_x_cond_z
                self.q_dist = StudentT(df=1)
                self._q_prob = self.q_dist.log_prob(value=self.X).exp()
                self.w_adjusted = self._q_prob / self.w
                self.w_adjusted_sum = self.w_adjusted.sum()
            else:
                self.w_adjusted = self.cme_x_given_z()
                self.w_adjusted_sum = self.w_adjusted.sum()
            self.w_adjusted_current = copy.deepcopy(self.w_adjusted)

    def calculate_statistic(self):
        with torch.no_grad():
            X_ker = self.kernel_X(self.X)
            Y_ker = self.kernel_Y(self.Y)
            return (X_ker@self.H@Y_ker).sum()/self.n**2

    def permutation_calculate_weighted_statistic(self):
        idx = torch.randperm(self.n)
        with torch.no_grad():
            X_ker = self.kernel_X(self.X) * self.w_adjusted
            Y_ker = self.kernel_Y(self.Y)
            centered = X_ker[idx]@self.H
            return (centered@Y_ker.evaluate()).sum()/(self.w_adjusted_sum*self.n**2)

    def calculate_weighted_statistic(self):
        with torch.no_grad():
            X_ker = self.kernel_X(self.X) * self.w_adjusted
            Y_ker = self.kernel_Y(self.Y)
            centered = X_ker@self.H
            return (centered@Y_ker.evaluate()).sum()/(self.w_adjusted_sum*self.n**2)










