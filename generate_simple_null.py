import torch
from torch.distributions import Bernoulli,Normal
import os
import matplotlib.pyplot as plt


def generate_simple_null(alp,new_dirname,seed,d,n,null_case):
    torch.random.manual_seed(seed)
    z_dist = Normal(0, 1)
    z_samples = z_dist.sample((n, d))
    sig_z = torch.sigmoid(z_samples)
    sig_z_neg = torch.sigmoid(-z_samples)
    x_dist = Bernoulli(probs=sig_z)
    x_samples = x_dist.sample(())
    if null_case:
        y_dist = Normal(z_samples * alp, 0.05 ** 0.5)
    else:
        y_dist = Normal((2 * x_samples - 1) * z_samples.abs() * alp, 0.05 ** 0.5)

    y_samples = y_dist.sample(())
    if seed == 1:
        plt_y_marg = y_samples.numpy()
        plt.hist(plt_y_marg, bins=100)
        plt.savefig(new_dirname + '/y_marg.png')
        plt.clf()

    w = torch.where(x_samples == 1, 1 / (2 * sig_z), 1 / (2 * sig_z_neg))
    w = w.prod(dim=1)
    return x_samples,y_samples,z_samples,w

if __name__ == '__main__':
    n=10000
    seeds=100
    for d in [1,3,50]:
        for null_case in [True,False]:
            for alp in [0.0,2*1e-2,4*1e-2,6*1e-2,8*1e-2,1e-1]:
                new_dirname = f'do_null_univariate_alp={alp}_null={null_case}_d={d}'
                if not os.path.exists(new_dirname):
                    os.makedirs(new_dirname)
                for i in range(seeds):
                    x_samples, y_samples, z_samples, w = generate_simple_null(alp,new_dirname=new_dirname,seed=i,d=d,n=n,null_case=null_case)
                    torch.save((x_samples,y_samples,z_samples,w.squeeze()),new_dirname+'/'+f'data_seed={i}.pt')


    # w_i = 1/sigma(z_i) if X==1 else 1-1/sigma(z_i).

