from kgformula.kernels import *
import torch
import numpy as np
kernel_base=Kernel()

class old_statistic():
    def __init__(self,X,Y,Z,X_te,Y_te,Z_te ,permutations = 250,device='cuda:0'):
        self.iterations = permutations
        self.X=X
        self.Y=Y
        self.Z=Z
        self.X_te=X_te
        self.Y_te=Y_te
        self.Z_te=Z_te
        self.n_tr = self.X.shape[0]
        self.n_te = self.Y_te.shape[0]
        ls_x = self.get_median_ls(X)
        ls_y = self.get_median_ls(Y)
        ls_z = self.get_median_ls(Z)
        self.kx=RBFKernel()
        self.kx.ls = ls_x
        self.ky=RBFKernel()
        self.ky.ls = ls_y
        self.kz=RBFKernel()
        self.kz.ls = ls_z
        self.KX = self.kx.forward(self.X)
        self.KZ = self.kz.forward(self.Z)
        self.KY = self.ky.forward(self.Y_te)

        self.kz_sum = self.KZ.sum(dim=1)
        self.KX_test =self.kx.forward(self.X,self.X_te)
        reg = torch.eye(self.n_tr).to(self.KX.device)*1e-3
        solve_mat = self.KX*self.KZ
        solve_b = self.KX_test*self.kz_sum
        L = torch.linalg.cholesky(solve_mat+ reg)
        self.W = torch.cholesky_solve(solve_b,L)/self.n_tr
        self.centered_W = self.W - torch.mean(self.W,dim=1)
        self.W_mat = self.centered_W@self.centered_W.transpose()
        self.ref_stat = torch.sum(self.W_mat* self.KY).item()

    def get_permuted2d(self,ker):
        idx = torch.randperm(self.n_te)
        kernel_X = ker[:,idx]
        kernel_X = kernel_X[idx,:]
        return kernel_X,idx

    def permutation_test(self):
        null_vec = np.zeros(self.iterations)
        for i in range(self.iterations):
            perm_ker,idx = self.get_permuted2d(self.KY)
            null_vec[i] = torch.sum(perm_ker*self.W_mat).item()
        return null_vec, self.ref_stat

    def get_median_ls(self,X):
        with torch.no_grad():
            if X.shape[0]>5000:
                X = X[:5000,:]
            d = kernel_base.covar_dist(x1=X,x2=X)
            ret = torch.sqrt(torch.median(d[d >= 0]))
            return ret
