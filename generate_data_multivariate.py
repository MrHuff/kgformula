from kgformula.fixed_do_samplers import simulate_xyz_multivariate
from kgformula.test_statistics import  hsic_sanity_check_w
import torch
import os
if __name__ == '__main__':
    n = 1000
    seeds = 1000
    save_path = ''
    d_Z = 50
    yz = 0.5
    d_X = 3
    d_Y = 3
    b_z = 1.
    beta_xz = [0.0]+[round(b_z/(d_Z**2),5)]*(d_Z) #Some choices of theta and phi render w_true to be invalid. Why? We want do derive cases where w_true makes the test valid.
    #Both theta and phi controlls odds ratio and distribution of X...
    b_y =  0.5
    for beta_xy in [[0,b_y],[0,0]]:
        data_dir = f'beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n={n}_yz={yz}_beta_XZ={round(b_z/(d_Z**2),5)}'
        if not os.path.exists(f'./{data_dir}/'):
            os.mkdir(f'./{data_dir}/')
        for i in range(seeds):
            X,Y,Z,w = simulate_xyz_multivariate(n,oversamp=5,d_Z=d_Z,beta_xz=beta_xz,beta_xy=beta_xy,seed = i,yz=yz,d_X=d_X,d_Y=d_Y,phi=8,theta=8)
            if i==0:
                sanity_pval = hsic_sanity_check_w(w, X, Z, 1000)
                print(f'sanity_check: {sanity_pval}')
            torch.save((X,Y,Z,w),f'./{data_dir}/data_seed={i}.pt')



