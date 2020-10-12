from kgformula.fixed_do_samplers import simulate_xyz_multivariate
from kgformula.test_statistics import  hsic_sanity_check_w,hsic_test
import torch
import os
import time
from scipy.stats import kstest
from matplotlib import pyplot as plt
import numpy as np
from kgformula.utils import ecdf
def generate_sensible_variables(d_Z,b_Z,const=0):
    if d_Z==1:
        variables = [b_Z]
    else:
        bucket_size = d_Z // 3 if d_Z > 2 else 1
        variables = [0]*d_Z
        a = round(b_Z/(d_Z**2),5)
        variables[0:bucket_size]= [a for i in range(0,bucket_size)]
        variables[bucket_size:2*bucket_size]=[a/10. for i in range(bucket_size,2*bucket_size)]
    return [const] + variables


if __name__ == '__main__': #This is incorrectly generated...
        seeds = 100
        yz = 0.5
        d_X = 3 #Try 2 and 1
        d_Y = 3 #Try 1
        for d_Z in [3,50]: #50,3
            for b_z in [0.0,1e-2,0.1,0.5,1]: #,1e-3,1e-2,0.05,0.1,0.25,0.5,1
                b_z= (d_Z**2)*b_z
                beta_xz = generate_sensible_variables(d_Z,b_z,const=0)#What if X and Z indepent -> should be uniform, should sanity check that this actully is well behaved for all d_Z.
                #Try different beta configs, i.e.
                b_y =  1.0
                theta = 2.0 #2,4
                phi = 2.0 #choose multiples instead of powers...
                for n in [10000]:
                    for beta_xy in [[0,0]]:
                        for q_fac in [0.4,0.2,0.01,0.6,0.8,0.9,1.0]:
                            data_dir = f"q={q_fac}_mv_{seeds}/beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n={n}_yz={yz}_beta_XZ={round(b_z/(d_Z**2),3)}_theta={theta}_phi={round(phi,2)}"
                            if not os.path.exists(f'./{data_dir}/'):
                                os.makedirs(f'./{data_dir}/')
                            for i in range(seeds):
                                X,Y,Z,X_q,w,w_q = simulate_xyz_multivariate(n,oversamp=5,d_Z=d_Z,beta_xz=beta_xz,beta_xy=beta_xy,seed = i,yz=yz,d_X=d_X,d_Y=d_Y,phi=phi,theta=theta,q_fac=q_fac)
                                torch.save((X,Y,Z,X_q,w,w_q),f'./{data_dir}/data_seed={i}.pt')
                                beta_xz = torch.Tensor(beta_xz)
                                if i == 0:
                                    with torch.no_grad():
                                        e_xz = torch.cat([torch.ones(*(X.shape[0], 1)), Z],
                                                         dim=1) @ beta_xz  # XZ dependence
                                        sig_xxz = theta
                                        for d in range(d_X):
                                            sample_X = (X[:,d] - e_xz).squeeze().numpy()  # *sig_xxz
                                            stat, pval = kstest(sample_X, 'norm', (0, sig_xxz))
                                            # print(sig_xxz)
                                            print(f'KS-stat: {stat}, pval: {pval}')
                                            print(w.std())
                                            print(w.max())
                                            print(w.min())

                                    p_val = hsic_test(X, Z, 100)
                                    sanity_pval = hsic_sanity_check_w(w, X, Z, 100)
                                    print(f'HSIC X Z: {p_val}')
                                    print(f'sanity_check_w : {sanity_pval}')
                                    plt.hist(w_q, bins=250)
                                    plt.savefig(f'./{data_dir}/w_q.png')
                                    plt.clf()
                                    plt.hist(w, bins=250)
                                    plt.savefig(f'./{data_dir}/w.png')
                                    plt.clf()


