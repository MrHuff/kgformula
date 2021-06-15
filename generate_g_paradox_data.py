import torch
import numpy as np
import os
import shutil
def sample_cont_A(A_minus_1,L_k):
    A = torch.randn_like(L_k)*25+80+0.1*A_minus_1+30*L_k-0.05*A_minus_1*L_k
    return torch.clip(A,0,200)
def sample_bin_A(A_minus_1,L_k):
    p = torch.sigmoid(-1.25+A_minus_1+L_k+A_minus_1*L_k)
    d = torch.distributions.Bernoulli(probs=p)
    return d.sample(sample_shape=())

def sim_g_paradox(n_size,K,binary_flag=False,seed=1):
    torch.manual_seed(seed)
    if binary_flag:
        alpha_0 = 1
        alpha_1 = -0.015
        alpha_2 = 1
        alpha_3 = 0.015
    else:
        alpha_0=0
        alpha_1=-2.5
        alpha_2=1
        alpha_3=2.5
    U = torch.rand(n_size)
    Y = torch.clip(torch.randn(n_size)*50 + U*300+350,0,1000).unsqueeze(-1)
    A=0
    Z = []
    X = []
    for k in range(K):
        A_minus_1 = A
        p = torch.sigmoid(alpha_0+alpha_1*A_minus_1+alpha_2*U+alpha_3*U*A_minus_1)
        d = torch.distributions.Bernoulli(probs=p)
        L_k = d.sample(sample_shape=())
        if binary_flag:
            A= sample_bin_A(A_minus_1=A_minus_1,L_k=L_k)
        else:
            A= sample_cont_A(A_minus_1=A_minus_1,L_k=L_k)
        Z.append(L_k)
        X.append(A)
    X = torch.stack(X,dim=1)
    Z = torch.stack(Z,dim=1)
    return X,Y,Z

def generate_paradox_data(n_size,K,binary_flag,seeds):
    fold_name = f'g_paradox_data_{n_size}_{K}_{binary_flag}_{seeds}'
    if not os.path.exists(fold_name):
        os.makedirs(fold_name)
    else:
        shutil.rmtree(fold_name)
        os.makedirs(fold_name)
    for i in range(seeds):
        X, Y, Z = sim_g_paradox(n_size, K=K, binary_flag=binary_flag)

        w_dummy = torch.ones_like(Y)
        torch.save((X,Y,Z,w_dummy),f'{fold_name}/data_seed={i}.pt')


if __name__ == '__main__':
    generate_paradox_data(10000,10,True,100)
    generate_paradox_data(10000,10,False,100)
