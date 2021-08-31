from generate_simple_null import generate_simple_null
from generate_data_multivariate import *
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
    jobs = []
    yz = [0.5, 0.0]  # Counter example
    xy_const = 0.0  # GCM breaker
    fam_z = 1
    fam_y = 1
    fam_x = [1, 1]
    folder_name = f'do_null_mix_new'
    sanity = False
    null=None

        # for d_X, d_Y, d_Z, theta, phi in zip([d], [d], [ d], [4.0],
        #                                      [2.0]):  # 50,3
    # for d_X, d_Y, d_Z, theta, phi in zip([2, 4, 6, 8], [2, 4, 6, 8], [2, 3, 15, 50], [2.0, 4.0, 16.0, 16.0],
    #                                      [2.0, 2.0, 2.0, 2.0]):
    # for d_X, d_Y, d_Z, theta, phi in zip([6, 8], [ 6, 8], [15, 50], [ 16.0, 16.0],
    #                                      [ 2.0, 2.0]):

    # theta_vec = [16.0]
    # phi_vec = [2.0]
    # d_X=[8]
    # d_Y=[8]
    # d_Z=[50]
    # els = list(product(*[d_X,d_Y,d_Z,theta_vec,phi_vec]))
    # for d_X,d_Y,d_Z, theta,phi in els: #Screwed up rip

    for d_X, d_Y, d_Z, theta, phi in zip([2, 4, 6, 8], [2, 4, 6, 8], [2, 3, 15, 50], [2.0, 4.0, 16.0, 16.0],
                                         [2.0, 2.0, 2.0, 2.0]):
        for b_z in [0.5]:  # ,1e-3,1e-2,0.05,0.1,0.25,0.5,1
            b_z = (d_Z ** 2) * b_z
            beta_xz = generate_sensible_variables(d_Z, b_z,const=0)  # What if X and Z indepent -> should be uniform, should sanity check that this actully is well behaved for all d_Z.
            for n in [10000]:
                # for beta_xy in [[0, 0.0], [0, 0.1], [0, 0.2], [0, 0.3], [0, 0.4], [0, 0.5]]:
                for beta_xy in [[0,0.0],[0,0.05],[0,0.1],[0,0.15],[0,0.2]]:
                    data_dir = f"{folder_name}_{seeds}/beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n={n}_yz={yz}_beta_XZ={round(b_z / (d_Z ** 2), 3)}_theta={theta}_phi={round(phi, 2)}"
                    if not os.path.exists(f'./{data_dir}/'):
                        os.makedirs(f'./{data_dir}/')
                    for i in range(seeds):
                        jobs.append(
                            [data_dir, n, d_Z, beta_xz, beta_xy, i, yz, d_X, d_Y, phi, theta, fam_x, fam_z, fam_y, sanity,null])
    # for el in jobs:
    #     gen_job_wrapper(el)
    import torch.multiprocessing as mp
    pool = mp.Pool(mp.cpu_count()-4)
    pool.map(gen_job_wrapper, [el for el in jobs])
#


