import torch
from kgformula.utils import calculate_power,hypothesis_acceptance
import gpytorch
from torch.distributions.studentT import StudentT
import copy
class weighted_stat():
    def __init__(self,X,Y,Z,w,do_null = True,reg_lambda=1e-3):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.n = X.shape[0]
        self.do_null=do_null
        self.W= w@w.t()
        self.kernel_base = gpytorch.kernels.Kernel()
        self.reg_lambda = reg_lambda
        self.H = torch.ones(*(self.n,self.n))*(1-1/self.n)
        for name,data in zip(['kernel_X','kernel_Y','kernel_Z'],[X,Y,Z]):
            self.kernel_ls_init(name,data)

    def kernel_ls_init(self,name,data):
        ker = gpytorch.kernels.RBFKernel()
        ls = self.get_median_ls(data)
        ker._set_lengthscale(ls)
        setattr(self,name,ker)

    def get_median_ls(self,X):
        with torch.no_grad():
            d = self.kernel_base.covar_dist(x1=X,x2=X)
            return torch.sqrt(torch.median(d[d > 0]))

    def permutation_calculate_weighted_statistic(self):
        idx = torch.randperm(self.n)
        with torch.no_grad():
            X_ker = self.kernel_X(self.X)
            Y_ker = self.kernel_Y(self.Y[idx])
            A = self.H@gpytorch.matmul(X_ker,self.H)*(self.H@gpytorch.matmul(Y_ker,self.H))
            return A.sum()

    def calculate_weighted_statistic(self):
        with torch.no_grad():
            X_ker = self.kernel_X(self.X)
            Y_ker = self.kernel_Y(self.Y)
            A = self.H@gpytorch.matmul(X_ker,self.H)*(self.H@gpytorch.matmul(Y_ker,self.H))
            return A.sum()










