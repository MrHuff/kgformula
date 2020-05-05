from kgformula.fixed_do_samplers import simulate_xyz_multivariate
import torch
import os
if __name__ == '__main__':
    n = 1000
    seeds = 100
    save_path = ''
    d_Z = 3
    beta_xz = [0.0]+[0.25]*(d_Z) #SIMULATION IS INCORRECT DEBUG TIME
    yz = 0.5
    for beta_xy in [[0,0.0],[0,0.5]]:
        data_dir = f'beta_xy={beta_xy}_d_Z={d_Z}_n={n}_yz={yz}_beta_XZ={beta_xz}'
        if not os.path.exists(f'./{data_dir}/'):
            os.mkdir(f'./{data_dir}/')
        for i in range(seeds):
            X,Y,Z,w = simulate_xyz_multivariate(n,oversamp=50,d_Z=d_Z,beta_xz=beta_xz,beta_xy=beta_xy,seed = i,yz=yz)
            torch.save((X,Y,Z,w),f'./{data_dir}/data_seed={i}.pt')



