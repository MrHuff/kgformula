from kgformula.fixed_do_samplers import sample_naive_multivariate
import torch
import os
if __name__ == '__main__':
    n = 1000
    seeds = 1000
    save_path = ''
    d_X = 3
    d_Y = 2
    d_Z = 5
    beta_xz = 0.5
    for beta_xy in [0,0.5]:
        data_dir = f'beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n={n}_beta_XZ={beta_xz}'
        if not os.path.exists(f'./{data_dir}/'):
            os.mkdir(f'./{data_dir}/')
        for i in range(seeds):
            X,Y,Z,w = sample_naive_multivariate(n,d_X=d_X,d_Y=d_Y,d_Z=d_Z,beta_xz=beta_xz,beta_xy=beta_xy,seed = i)
            torch.save((X,Y,Z,w),f'./{data_dir}/data_seed={i}.pt')

