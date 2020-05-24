from kgformula.fixed_do_samplers import simulate_xyz_multivariate
from kgformula.test_statistics import  hsic_sanity_check_w,hsic_test
import torch
import os
def generate_sensible_variables(d_Z,b_Z,const=0):
    bucket_size = d_Z//3
    print(bucket_size)
    variables = [0]*d_Z
    a = round(b_Z/(d_Z**2),5)
    variables[0:bucket_size]= [a for i in range(0,bucket_size)]
    variables[bucket_size:2*bucket_size]=[a/10. for i in range(bucket_size,2*bucket_size)]
    return [const] + variables


if __name__ == '__main__':
    for d_Z in [3]:
        seeds = 100
        save_path = ''
        yz = 0.5
        d_X = 3
        d_Y = 3
        b_z = 0.3 #In higher dimensions, the take seems to be that one must be careful with how much dependence is created.
        beta_xz = generate_sensible_variables(d_Z,b_z,const=0)#What if X and Z indepent -> should be uniform, should sanity check that this actully is well behaved for all d_Z.
        #1. Simulate realistic epidemilogy dataset witch partial/weak dependence between X and Z
        #2. Simluate sanity check for well-behavedness of NCE estimate when X \perp Z.
        #Some choices of theta and phi render w_true to be invalid. Why? We want do derive cases where w_true makes the test valid.
        #Both theta and phi controlls odds ratio and distribution of X...
        b_y =  1.0
        phi = 2*2**0.5
        theta = 8
        for n in [100,1000,10000]:
            for beta_xy in [[0,b_y],[0,0]]:
                data_dir = f'beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n={n}_yz={yz}_beta_XZ={round(b_z/(d_Z**2),5)}_theta={theta}_phi={round(phi,2)}'
                if not os.path.exists(f'./{data_dir}/'):
                    os.mkdir(f'./{data_dir}/')
                for i in range(seeds):
                    X,Y,Z,w = simulate_xyz_multivariate(n,oversamp=5,d_Z=d_Z,beta_xz=beta_xz,beta_xy=beta_xy,seed = i,yz=yz,d_X=d_X,d_Y=d_Y,phi=phi,theta=theta)
                    if i==0:
                        p_val = hsic_test(X, Z, 1000)
                        sanity_pval = hsic_sanity_check_w(w, X, Z, 1000)
                        print(f'HSIC X Z: {p_val}')
                        print(f'sanity_check_w : {sanity_pval}')
                    torch.save((X,Y,Z,w),f'./{data_dir}/data_seed={i}.pt')



