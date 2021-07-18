from generate_simple_null import generate_simple_null
from generate_data_multivariate import *


if __name__ == '__main__':
    seeds = 100
    jobs = []
    yz = [0.5, 0.0]  # Counter example
    xy_const = 0.0  # GCM breaker
    fam_z = 1
    fam_y = 1
    fam_x = [1, 1]
    folder_name = f'do_null_mix_100'
    sanity = False
    d=3
    # for d_X, d_Y, d_Z, theta, phi in zip([1, 3, 3, 3], [1, 3, 3, 3], [1, 3, 15, 50], [2.0, 4.0, 8.0, 16.0],
    #                                      [2.0, 2.0, 2.0, 2.0]):  # 50,3
    for null in [True,False]:
        for d_X, d_Y, d_Z, theta, phi in zip([d], [d], [ d], [4.0],
                                             [2.0]):  # 50,3
            for b_z in [0.5]:  # ,1e-3,1e-2,0.05,0.1,0.25,0.5,1
                b_z = (d_Z ** 2) * b_z
                beta_xz = generate_sensible_variables(d_Z, b_z,
                                                      const=0)  # What if X and Z indepent -> should be uniform, should sanity check that this actully is well behaved for all d_Z.
                for n in [10000]:
                    # for beta_xy in [[0, 0.0], [0, 0.1], [0, 0.2], [0, 0.3], [0, 0.4], [0, 0.5]]:
                    if null:
                        beta_xy = [0,0.0]
                    else:
                        beta_xy = [0,0.2]
                    data_dir = f"{folder_name}_{seeds}/beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n={n}_yz={yz}_beta_XZ={round(b_z / (d_Z ** 2), 3)}_theta={theta}_phi={round(phi, 2)}"
                    if not os.path.exists(f'./{data_dir}/'):
                        os.makedirs(f'./{data_dir}/')
                    for i in range(seeds):
                        jobs.append(
                            [data_dir, n, d_Z, beta_xz, beta_xy, i, yz, d_X, d_Y, phi, theta, fam_x, fam_z, fam_y, sanity])
        for el in jobs:
            X_cont,Y_cont,Z_cont,w_cont = generate_data_simple(el)
            X_bin,Y_bin,Z_bin, w_bin = generate_simple_null(alp=0.05,new_dirname=data_dir,seed=el[5],d=d,n=el[1],null_case=null)
            # X = torch.cat([X_bin],dim=1)
            X = torch.cat([X_bin,X_cont],dim=1)
            # X = torch.cat([X_cont],dim=1)
            # Y = torch.cat([Y_cont],dim=1)
            # Y = torch.cat([Y_bin],dim=1)
            Y = torch.cat([Y_bin,Y_cont],dim=1)
            Z = torch.cat([Z_cont,Z_bin],dim=1)
            torch.save((X,Y,Z,w_cont.squeeze()),data_dir+'/'+f'data_seed={el[5]}.pt')


