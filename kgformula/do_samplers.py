import numpy as np
import torch
from torch.distributions.exponential import Exponential
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.studentT import StudentT
import time
import math
def gaussLOR(x,y,rho=0.5):
    return rho * x * y / (1 - rho**2)

def wrapper(rho):
    def tmp(x,y):
      return rho * x * y / (1 - rho ** 2)
    return tmp
def gaussLOR_cond(x,y,z,beta):
    assert(z.shape[0]==beta.shape[0])
    rho = torch.tanh(torch.mm(z,beta))
    return  rho*x*y/(1-rho**2)

def swaps(x, y,N=500,log_or = gaussLOR):
    assert(x.shape[0]==y.shape[0])
    exp = Exponential(rate=1)
    n = x.shape[0]
    for i in range(N):
        wh = np.random.choice(n, 2, replace=False)
        xs = x[wh]
        ys = y[wh]
        log_alpha = -log_or(xs[0,],ys[0,]) - log_or(xs[1,],ys[1,]) + log_or(xs[0,],ys[1,]) + log_or(xs[1,],ys[0,])
        if -exp.sample((1,1))<log_alpha:
            x[wh[0],] = xs[1]
            x[wh[1],] = xs[0]
    return x,y

# def parallel_swaps(x, y,N=500,log_or = gaussLOR):
#     assert(x.shape[0]==y.shape[0])
#     exp = Exponential(rate=1)
#     n = x.shape[0]
#     exp_samples = -exp.rsample(N)
#
#     wh = np.empty((N,2))
#     for i in range(N):
#         wh[i,]=np.random.choice(n, 2, replace=False)
#
#     xs = x[wh]
#     ys = y[wh]
#
#     for i in range(N):
#         wh = np.random.choice(n, 2, replace=False)
#         xs = x[wh]
#         ys = y[wh]
#         log_alpha = -log_or(xs[0,],ys[0,]) - log_or(xs[1,],ys[1,]) + log_or(xs[0,],ys[1,]) + log_or(xs[1,],ys[0,])
#         if -exp.rsample(1)<log_alpha:
#             flip = np.flip(wh)
#             x[wh,] = x[flip,]
#     return x,y

def genBiv(n,x_mar=torch.randn,y_mar=torch.randn,log_or=gaussLOR,margin_indep =True,z=0,y_cond_z = lambda x: 0.6*x,cuda=False,N=0):

    if N==0 or N<n:
        N=10*n

    if margin_indep:
        x = x_mar(n)
        y = y_mar(n)
    else:
        x = x_mar(n)
        y = y_cond_z(z) + torch.tensor(0.1)*y_mar(n)

    if cuda:
        x = x.cuda()
        y = y.cuda()

    X,Y = swaps(x,y,N,log_or)
    return X, Y

def do_distribution(n,dist='ber',rho_xy=0,rho_xz=0.3,get_p_x_cond_z=False):
    with torch.no_grad():
        try_n = 4*n
        if dist =='t':
            dist = StudentT(df=3)
            x = dist.sample((try_n,1))+1
        if dist == 'ber':
            dist = Bernoulli(probs=0.5)
            x = dist.sample((try_n,1))
        y = torch.zeros_like(x)
        z = torch.zeros_like(x)

        k=10 #adjust to speed things up

        for i in range(try_n):
            rho_i = torch.clamp(torch.tanh(x[i]),min = -1+1e-9,max = 1-1e-9)
            tmp_Z,tmp_Y = genBiv(k,y_mar = lambda p: torch.randn(p)+x[i]*rho_xy,log_or=wrapper(rho_i))
            wh = np.random.choice(k,1)
            z[i] = tmp_Z[wh]
            y[i] = tmp_Y[wh]

        top = torch.exp(-( x -(torch.tensor(rho_xz*z+1)) )**2/2)*1/torch.sqrt(torch.tensor(2*math.pi)) #pdf of normal distribution
        bottom = dist.probs #pdf of bernoulli, its just 0.5
        wts = top/bottom
        wts = wts/torch.max(wts)

        keep = torch.rand_like(x)<wts # Compare to a uniform distribution (rejection rampling step)
        x = x[keep].unsqueeze(-1)
        y = y[keep].unsqueeze(-1)
        z = z[keep].unsqueeze(-1)
        if get_p_x_cond_z:
            return x,y,z,top[keep].unsqueeze(-1)
        else:
            return x,y,z

# if __name__ == '__main__':
#     print(do_distribution(50))


