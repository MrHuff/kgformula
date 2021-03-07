from generate_data_multivariate_non_gaussian_GCM_breaker import *
from kgformula.utils import *


if __name__ == '__main__':
    seeds = 100
    yz = [0.5,0.0] #GCM breaker - coplua
    xy_const = 0.0 #GCM breaker
    fam_z = 3
    fam_y= 3
    fam_x =[3,3]
    folder_name = f'exp_hsic_break_{seeds}'
    jobs = []
    for d_X,d_Y,d_Z, theta,phi in zip( [1],[1],[1],[0.1],[0.9]): #50,3
        for b_z in [0.5 ]:  # ,1e-3,1e-2,0.05,0.1,0.25,0.5,1
            b_z = (d_Z ** 2) * b_z
            beta_xz = generate_sensible_variables(d_Z, b_z, const=0)  # What if X and Z indepent -> should be uniform, should sanity check that this actully is well behaved for all d_Z.
            # Try different beta configs, i.e.
            # Smaller theta, but not too small. Acceptable SNR...
            # Somewhere in between for d_Z
            # High SNR for X to Z makes it harder
            # "perfect" balance between X and Z make it just hard enough so TRE and NCE_Q can get the job done.
            # for n in [1000,5000,10000]:

            for n in [10000]:
                for beta_xy in [[xy_const, 0.0],[xy_const, 0.5]]:
                    data_dir = f"{folder_name}/beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n={n}_yz={yz}_beta_XZ={round(b_z / (d_Z ** 2), 3)}_theta={theta}_phi={round(phi, 2)}"
                    if not os.path.exists(f'./{data_dir}/'):
                        os.makedirs(f'./{data_dir}/')
                    for i in range(seeds):
                        # if not os.path.exists(f'./{data_dir}/data_seed={i}.pt'):
                        jobs.append([data_dir,n,d_Z,beta_xz,beta_xy,i,yz,d_X,d_Y,phi,theta,fam_x,fam_z,fam_y])
    import multiprocessing as mp
    pool = mp.Pool(mp.cpu_count())
    pool.map(multiprocess_wrapper, [row for row in jobs])
