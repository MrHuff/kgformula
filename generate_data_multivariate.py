from kgformula.fixed_do_samplers import simulate_xyz_multivariate,apply_qdist
from kgformula.test_statistics import  hsic_sanity_check_w,hsic_test
import torch
import os
from scipy.stats import kstest
from matplotlib import pyplot as plt
import matplotlib as mpl
from kgformula.utils import x_q_class
mpl.use('Agg')
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
def calc_ess(w):
    return (w.sum()**2)/(w**2).sum()


if __name__ == '__main__': #This is incorrectly generated...
    seeds = 100
    yz = 0.5
    # for d_X,d_Y,d_Z, theta,phi in zip( [3,3,3],[3,3,3],[3,15,50],[4.0,8.0,16.0],[2.0,2.0,2.0]): #50,3
    for d_X,d_Y,d_Z, theta,phi in zip( [3],[3],[3],[3.0],[2.0]): #50,3
        for b_z in [0.5]: #,1e-3,1e-2,0.05,0.1,0.25,0.5,1
            b_z= (d_Z**2)*b_z
            beta_xz = generate_sensible_variables(d_Z,b_z,const=0)#What if X and Z indepent -> should be uniform, should sanity check that this actully is well behaved for all d_Z.
            s_var = sum([el**2 for el in beta_xz])
            snr = s_var/theta**2
            print(snr)
            #Try different beta configs, i.e.
            # Smaller theta, but not too small. Acceptable SNR...
            # Somewhere in between for d_Z
            # High SNR for X to Z makes it harder
            # "perfect" balance between X and Z make it just hard enough so TRE and NCE_Q can get the job done.
            for n in [10000]:
                for beta_xy in [[0,0.0],[0,0.5],[0,0.25],[0,0.1]]:
                    data_dir = f"data_{seeds}/beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n={n}_yz={yz}_beta_XZ={round(b_z / (d_Z ** 2), 3)}_theta={theta}_phi={round(phi, 2)}"
                    if not os.path.exists(f'./{data_dir}/'):
                        os.makedirs(f'./{data_dir}/')
                    for i in range(seeds):
                        X,Y,Z,inv_w = simulate_xyz_multivariate(n,oversamp=5,d_Z=d_Z,beta_xz=beta_xz,beta_xy=beta_xy,seed = i,yz=yz,d_X=d_X,d_Y=d_Y,phi=phi,theta=theta)
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

                            p_val = hsic_test(X[0:1000,:], Z[0:1000,:], 250)
                            X_class = x_q_class(qdist=2, q_fac=1.0, X=X)
                            w = X_class.calc_w_q(inv_w)
                            sanity_pval = hsic_sanity_check_w(w[0:1000], X[0:1000, :], Z[0:1000, :], 250)
                            print(f'HSIC X Z: {p_val}')
                            print(f'sanity_check_w : {sanity_pval}')
                            plt.hist(w, bins=250)
                            plt.savefig(f'./{data_dir}/w.png')
                            plt.clf()
                            ess_list = []
                            for q in [1.0,0.75,0.5]:
                                X_class = x_q_class(qdist=2, q_fac=q, X=X)
                                w = X_class.calc_w_q(inv_w)
                                ess = calc_ess(w)
                                ess_list.append(ess.item())
                            max_ess = max(ess_list)
                            print('max_ess: ',max_ess)
                        torch.save((X,Y,Z,inv_w),f'./{data_dir}/data_seed={i}.pt')


