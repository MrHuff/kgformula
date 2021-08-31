from generate_data_multivariate_non_gaussian_GCM_breaker import *
from kgformula.utils import *
from itertools import *

def gen_job_wrapper(el):
    X, Y, Z, w_cont = generate_data_mixed(el)
    # X_bin, Y_bin, Z_bin, w_bin = generate_simple_null(alp=0 if el[-1] else 0.06, new_dirname=el[0], seed=el[5], d=el[7],
    #                                                   n=el[1], null_case=el[-1])
    # X = torch.cat([X_bin, X_cont], dim=1)
    # Y = torch.cat([Y_bin, Y_cont], dim=1)
    # Z = torch.cat([Z_cont, Z_bin], dim=1)
    torch.save((X, Y, Z, w_cont.squeeze() ), el[0] + '/' + f'data_seed={el[5]}.pt')


if __name__ == '__main__':
    seeds = 100
    jobs=[]
    yz = [0.5, 0.0]  # Counter example
    xy_const = 0.0  # GCM breaker
    fam_z = 1
    fam_y = 1
    fam_x = [1, 1]
    folder_name = f'ablation_100'
    sanity = False
    theta_vec = [2.0]
    phi_vec = [2.0]
    d_X=[2]
    d_Y=[2]
    d_Z=[2]
    els = list(product(*[d_X,d_Y,d_Z,theta_vec,phi_vec]))
    for d_X,d_Y,d_Z, theta,phi in els: #Screwed up rip
        for b_z in [0.5]:  # ,1e-3,1e-2,0.05,0.1,0.25,0.5,1
            b_z = (d_Z ** 2) * b_z
            beta_xz = generate_sensible_variables(d_Z, b_z, const=0)  # What if X and Z indepent -> should be uniform, should sanity check that this actully is well behaved for all d_Z.
            # for n in [1000,5000,10000]:

            for n in [10000]:
                for beta_xy in [[xy_const, 0.01],[xy_const, 0.03],[xy_const, 0.05],[xy_const, 0.07],[xy_const, 0.09]]:
                    data_dir = f"{folder_name}/beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n={n}_yz={yz}_beta_XZ={round(b_z / (d_Z ** 2), 3)}_theta={theta}_phi={round(phi, 2)}"
                    if not os.path.exists(f'./{data_dir}/'):
                        os.makedirs(f'./{data_dir}/')
                    for i in range(seeds):
                        # if not os.path.exists(f'./{data_dir}/data_seed={i}.pt'):
                        jobs.append([data_dir,n,d_Z,beta_xz,beta_xy,i,yz,d_X,d_Y,phi,theta,fam_x,fam_z,fam_y,sanity])
    import multiprocessing as mp
    pool = mp.Pool(mp.cpu_count())
    pool.map(multiprocess_wrapper, [row for row in jobs])
    # for el in jobs:
    #     multiprocess_wrapper(el)