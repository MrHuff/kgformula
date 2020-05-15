from kgformula.fixed_do_samplers import simulate_xyz_multivariate
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
    beta_xz = [0.0]+[1/d_Z]*(d_Z) #SIMULATION IS INCORRECT DEBUG TIME
    b_y = torch.eye(d_X)*0.5

    for beta_xy in [[0,0.0],[0,b_y]]:
        data_dir = f'beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n={n}_yz={yz}_beta_XZ={beta_xz[:5]}'
        if not os.path.exists(f'./{data_dir}/'):
            os.mkdir(f'./{data_dir}/')
        for i in range(seeds):
            X,Y,Z,w = simulate_xyz_multivariate(n,oversamp=2,d_Z=d_Z,beta_xz=beta_xz,beta_xy=beta_xy,seed = i,yz=yz,d_X=d_X,d_Y=d_Y,phi=2**d_X,theta=2**d_X)
            torch.save((X,Y,Z,w),f'./{data_dir}/data_seed={i}.pt')



