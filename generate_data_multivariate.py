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
    bucket_size = d_Z//3
    print(bucket_size)
    variables = [0]*d_Z
    a = round(b_Z/(d_Z**2),5)
    variables[0:bucket_size]= [a for i in range(0,bucket_size)]
    variables[bucket_size:2*bucket_size]=[a/10. for i in range(bucket_size,2*bucket_size)]
    return [const] + variables


if __name__ == '__main__':
        seeds = 100
        yz = 0.5
        d_X = 3 #Try 2 and 1
        d_Y = 3 #Try 1
        #50, b_z=1.0, 2*sqrt(2),4
        #3, b_z = 0.3, 2*sqrt(2),8
        for d_Z,b_z in zip([3,50],[4.5,25]): #50,3
            #Test d_Z 1,2
             #25,4.5 #In higher dimensions, the take seems to be that one must be careful with how much dependence is created.
            beta_xz = generate_sensible_variables(d_Z,b_z,const=0)#What if X and Z indepent -> should be uniform, should sanity check that this actully is well behaved for all d_Z.
            #1. Simulate realistic epidemilogy dataset witch partial/weak dependence between X and Z
            #2. Simluate sanity check for well-behavedness of NCE estimate when X \perp Z.
            #Some choices of theta and phi render w_true to be invalid. Why? We want do derive cases where w_true makes the test valid.
            #Both theta and phi controlls odds ratio and distribution of X...
            b_y =  1.0
            theta = 6 #2,4
            phi = theta/1.75 #choose multiples instead of powers...
            for n in [10000]:
                for beta_xy in [[0,b_y],[0,0]]:
                    for q_fac in [1.25,1.5]:
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
                                    # plt.hist(w.numpy(),bins=25)
                                    # plt.show()
                                    # plt.clf()
                                    sig_xxz = phi
                                    for d in range(d_X):
                                        sample_X = (X[:,d] - e_xz).squeeze().numpy()  # *sig_xxz
                                        # ref = np.random.randn(n)*sig_xxz
                                        # x_ref,y_ref = ecdf(ref)
                                        # x,y = ecdf(sample_X)
                                        # plt.scatter(x=x, y=y)
                                        # plt.scatter(x=x_ref, y=y_ref)
                                        # plt.show()
                                        # plt.clf()
                                        stat, pval = kstest(sample_X, 'norm', (0, sig_xxz))
                                        # print(sig_xxz)
                                        print(f'KS-stat: {stat}, pval: {pval}')
                                        print(w.std())
                                        print(w.max())
                                        print(w.min())
                                        p_val = hsic_test(X, Z, 1000)
                                        sanity_pval = hsic_sanity_check_w(w, X, Z, 1000)
                                        print(f'HSIC X Z: {p_val}')
                                        print(f'sanity_check_w : {sanity_pval}')
                                        time.sleep(0.5)
                                p_val = hsic_test(X, Z, 1000)
                                sanity_pval = hsic_sanity_check_w(w, X, Z, 1000)
                                print(f'HSIC X Z: {p_val}')
                                print(f'sanity_check_w : {sanity_pval}')



