import torch
import os
data_dir ='univariate_100_seeds/univariate_test'
n=10000
seeds = 100

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
for i in range(seeds):
    torch.manual_seed(i)
    dat = torch.randn(*(n,4))
    X,Y,Z,X_q = dat.unbind(1)
    X = X.unsqueeze(-1)
    X_q = X_q.unsqueeze(-1)
    Y = Y.unsqueeze(-1)
    Z = Z.unsqueeze(-1)
    w = torch.rand(n)
    w_q = torch.rand(n)
    torch.save((X, Y, Z, X_q, w, w_q), f'./{data_dir}/data_seed={i}.pt')

