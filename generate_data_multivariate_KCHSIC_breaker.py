from generate_data_multivariate import *
from kgformula.utils import *
import seaborn as sns
import sklearn


if __name__ == '__main__':
    seeds = 100
    jobs=[]
    yz = [0.5, 0.0]  # Counter example
    xy_const = 0.0  # GCM breaker
    fam_z = 1
    fam_y = 1
    fam_x = [1, 1]
    folder_name = f'kchsic_break'
    sanity = False
    # for d_X,d_Y,d_Z, theta,phi in zip( [1,3,3,3],[1,3,3,3],[1,3,15,50],[2.0,4.0,8.0,16.0],[2.0,2.0,2.0,2.0]): #50,3
    for d_X,d_Y,d_Z, theta,phi in zip( [1],[1],[1],[4.0],[2.0]): #50,3
        for b_z in [0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5]: #,1e-3,1e-2,0.05,0.1,0.25,0.5,1
            b_z= (d_Z**2)*b_z
            beta_xz = generate_sensible_variables(d_Z,b_z,const=0)#What if X and Z indepent -> should be uniform, should sanity check that this actully is well behaved for all d_Z.
            for n in [10000]:
                for beta_xy in [[0,0.0]]:
                    data_dir = f"{folder_name}_{seeds}/beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n={n}_yz={yz}_beta_XZ={round(b_z / (d_Z ** 2), 3)}_theta={theta}_phi={round(phi, 2)}"
                    if not os.path.exists(f'./{data_dir}/'):
                        os.makedirs(f'./{data_dir}/')
                    for i in range(seeds):
                        jobs.append([data_dir,n,d_Z,beta_xz,beta_xy,i,yz,d_X,d_Y,phi,theta,fam_x,fam_z,fam_y,sanity])
    import torch.multiprocessing as mp
    pool = mp.Pool(mp.cpu_count())
    pool.map(multiprocess_wrapper, [row for row in jobs])
    # for el in jobs:
    #     multiprocess_wrapper(el)