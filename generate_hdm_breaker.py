from generate_data_multivariate import calc_ess,calc_snr,sanity_check_marginal_z,pairs_plot,conditional_dependence_plot_1d
from kgformula.fixed_do_samplers import U_func,simulate_xyz_multivariate_U_shape
from kgformula.utils import *
import seaborn as sns
import sklearn
from itertools import *


def sanity_check_marginal_y(X,Y,beta_xy,fam_y,data_dir):
    a = beta_xy[0]
    b = beta_xy[1]
    p = U_func(fam_y,X,a,b)
    cdf_y = p.cdf(Y)
    stat, pval = kstest(cdf_y.numpy().squeeze(), 'uniform')
    plt.hist(cdf_y.numpy(), bins=50,density=True)
    plt.savefig(f'./{data_dir}/Y_marg_histogram.png')
    plt.clf()
    print(f'pval Y-marginal:{pval}')
def gen_data_and_sanity_U_shape(data_dir,n,d_Z,beta_xz,beta_xy,i,yz,d_X,d_Y,phi,theta,fam_x,fam_z,fam_y,sanity):
    try:

        X, Y, Z, inv_w = simulate_xyz_multivariate_U_shape(n, oversamp=10, d_Z=d_Z, beta_xz=beta_xz,
                                                   beta_xy=beta_xy, seed=i, yz=yz, d_X=d_X, d_Y=d_Y,
                                                   phi=phi, theta=theta, fam_x=fam_x, fam_z=fam_z, fam_y=fam_y)
        if sanity:
            beta_xz = torch.Tensor(beta_xz)
            if i == 0:  # No way to sanity check...

                if d_X == 1:
                    sanity_check_marginal_z(Z, fam_z=fam_z,data_dir=data_dir)
                    sanity_check_marginal_y(X, Y, beta_xy, fam_y=fam_y,data_dir=data_dir)
                    pairs_plot(X, Y, Z,data_dir=data_dir)
                    conditional_dependence_plot_1d(X, Y, Z,data_dir=data_dir)
                _x_mu = torch.cat([torch.ones(*(X.shape[0], 1)), Z],
                                  dim=1) @ beta_xz  # XZ dependence
                for d in range(d_X):
                    plt.hist(X[:, d].squeeze().numpy(), bins=100)
                    plt.savefig(f'./{data_dir}/marginal_X.png')
                    plt.clf()

                    if fam_z == 1:
                        snr = calc_snr(beta_xz, theta)
                        test = torch.cat([torch.ones(*(X.shape[0], 1)), Z],
                                         dim=1) @ beta_xz  # XZ dependence
                        sig_xxz = theta
                        sample = (X[:, d] - test).squeeze().numpy()  # *sig_xxz
                        stat, pval = kstest(sample, 'norm', (0, sig_xxz))
                        print(f'KS test pval {pval}')
                        print('snr: ', snr)

                    plt.hist(test.squeeze().numpy(), bins=100, color='b', alpha=0.25)
                    plt.hist(sample.squeeze(), bins=100, color='r', alpha=0.25)
                    plt.savefig(f'./{data_dir}/test_vs_dist.png')
                    plt.clf()
        torch.save((X, Y, Z, inv_w), f'./{data_dir}/data_seed={i}.pt')
    except Exception as e:
        print(e)


def gen_job_wrapper(el):

    # try:
    # X, Y, Z, w_cont = gen_data_and_sanity(*el)
    gen_data_and_sanity_U_shape(*el)
    # torch.save((X, Y, Z, w_cont.squeeze() ), el[0] + '/' + f'data_seed={el[5]}.pt')
    # except Exception as e:
    #     print(e)
def generate_sensible_variables_old(d_Z,b_Z,const=0):
    if d_Z==1:
        variables = [b_Z]
    else:
        bucket_size = d_Z // 3 if d_Z > 2 else 1
        variables = [0]*d_Z
        a = round(b_Z/(d_Z**2),5)
        variables[0:bucket_size]= [a for i in range(0,bucket_size)]
        variables[bucket_size:2*bucket_size]=[a/10. for i in range(bucket_size,2*bucket_size)]
    return [const] + variables
if __name__ == '__main__':
    seeds = 1
    jobs=[]
    yz = [0.5, 0.0]  # Counter example
    xy_const = 0.0  # GCM breaker
    fam_z = 1
    sanity = True

    for fam_y in [1,4]:
        fam_x = [1, 1]
        folder_name = f'hdm_breaker_fam_y={fam_y}'

        for d_X,d_Y,d_Z, theta,phi in zip( [1,3],[1,3],[1,50],[2.0,16.0],[2.0,2.0]): #50,3: #Screwed up rip
            for b_z in [0.25,0.075]:  # ,1e-3,1e-2,0.05,0.1,0.25,0.5,1
                b_z = (d_Z ** 2) * b_z
                beta_xz = generate_sensible_variables_old(d_Z, b_z, const=0)  # What if X and Z indepent -> should be uniform, should sanity check that this actully is well behaved for all d_Z.
                for n in [10000]:
                    for beta_xy in [[0,0.0],[0,0.01],[0,0.02],[0,0.03],[0,0.04],[0,0.05]]:
                        data_dir = f"{folder_name}_{seeds}/beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n={n}_yz={yz}_beta_XZ={round(b_z / (d_Z ** 2), 3)}_theta={theta}_phi={round(phi, 2)}"
                        if not os.path.exists(f'./{data_dir}/'):
                            os.makedirs(f'./{data_dir}/')
                        for i in range(seeds):
                            jobs.append([data_dir,n,d_Z,beta_xz,beta_xy,i,yz,d_X,d_Y,phi,theta,fam_x,fam_z,fam_y,sanity])
    for el in jobs:
        gen_job_wrapper(el)
    # import torch.multiprocessing as mp
    # pool = mp.Pool(mp.cpu_count())
    # pool.map(gen_job_wrapper, [row for row in jobs])