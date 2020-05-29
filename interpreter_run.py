from kgformula.utils import simulation_object
import GPUtil
import torch.multiprocessing as mp
import math
import os

def run_jobs(seed_a,seed_b,theta,phi,beta_XZ_list,n_list,device,job_dir,runs=1):
    for beta_XZ in beta_XZ_list:
        for n in n_list:
            # h_0_str = f'ground_truth=H_0_y_a=0.0_y_b=0.0_z_a=0.0_z_b={beta_XZ}_cor=0.5_n={n}_seeds=100'
            # h_1_str = f'ground_truth=H_1_y_a=0.0_y_b=0.5_z_a=0.0_z_b={beta_XZ}_cor=0.5_n={n}_seeds=100'
            # h_0_str_mult_2 = f'beta_xy=[0, 0]_d_X=3_d_Y=3_d_Z=3_n={n}_yz=0.5_beta_XZ={beta_XZ}_theta={theta}_phi={phi}'
            # h_1_str_mult_2 = f'beta_xy=[0, 1.0]_d_X=3_d_Y=3_d_Z=3_n={n}_yz=0.5_beta_XZ={beta_XZ}_theta={theta}_phi={phi}'
            # h_0_str_mult_2_big = f'beta_xy=[0, 0]_d_X=3_d_Y=3_d_Z=50_n={n}_yz=0.5_beta_XZ={beta_XZ}'
            # h_1_str_mult_2_big = f'beta_xy=[0, 1.0]_d_X=3_d_Y=3_d_Z=50_n={n}_yz=0.5_beta_XZ={beta_XZ}'
            h_0_str = f'univariate_1k_seeds/ground_truth=H_0_y_a=0.0_y_b=0.0_z_a=0.0_z_b={beta_XZ}_cor=0.5_n={n}_seeds=1000'
            h_1_str = f'univariate_1k_seeds/ground_truth=H_1_y_a=0.0_y_b=0.5_z_a=0.0_z_b={beta_XZ}_cor=0.5_n={n}_seeds=1000'
            h_0_str_mult_2 = f'multivariate_1_K/beta_xy=[0, 0]_d_X=3_d_Y=3_d_Z=3_n={n}_yz=0.5_beta_XZ={beta_XZ}_theta={theta}_phi={phi}'
            h_1_str_mult_2 = f'multivariate_1_K/beta_xy=[0, 1.0]_d_X=3_d_Y=3_d_Z=3_n={n}_yz=0.5_beta_XZ={beta_XZ}_theta={theta}_phi={phi}'
            h_0_str_mult_2_big = f'multivariate_1_K/beta_xy=[0, 0]_d_X=3_d_Y=3_d_Z=50_n={n}_yz=0.5_beta_XZ={beta_XZ}_theta={theta}_phi={phi}'
            h_1_str_mult_2_big = f'multivariate_1_K/beta_xy=[0, 1.0]_d_X=3_d_Y=3_d_Z=50_n={n}_yz=0.5_beta_XZ={beta_XZ}_theta={theta}_phi={phi}'
            val_rate = max(1e-2, 10. / n)
            for h in [h_0_str]:  # zip([h_0_str_mult_2_big,h_1_str_mult_2_big],[seed_max,seed_max]):
                args = {
                    'device': device,
                    'job_dir':job_dir,
                    'data_dir': h,
                    'estimate': True,
                    'debug_plot': False,
                    'seeds_a': seed_a,
                    'seeds_b': seed_b,
                    'bootstrap_runs': 250,
                    'debug_generative_process': False,
                    'debug_d_Z': 3,
                    'est_params': {'lr': 1e-3,
                                   'max_its': 5000,
                                   'width': 512,
                                   'layers': int(math.log10(n)),
                                   'mixed': False,
                                   'bs_ratio': 10. / n,
                                   'kappa': 10,
                                   'val_rate': val_rate,
                                   'n_sample': 250,
                                   'criteria_limit': 0.05,
                                   'kill_counter': 10,
                                   'reg_lambda': 1e-2,
                                   'alpha': 0.5},
                    'estimator': 'classifier',
                    'runs': runs,
                    'cuda': True,
                }
                j = simulation_object(args)
                j.run()
                del j
if __name__ == '__main__':
    #problem children:
    #ground_truth=H_0_y_a=0.0_y_b=0.0_z_a=0.0_z_b=0.5_cor=0.5_n=10000_seeds=1000
    #beta_xy=[0, 0]_d_X=3_d_Y=3_d_Z=3_n=10000_yz=0.5_beta_XZ=0.03333_theta=8_phi=2.83
    #beta_xy=[0, 0]_d_X=3_d_Y=3_d_Z=3_n=100_yz=0.5_beta_XZ=0.03333_theta=8_phi=2.83

    beta_XZ_list = [0.5]
    n_list = [10000]
    theta = 8
    phi = 2.83
    seed_max = 1000
    cuda = True
    nr_of_gpus = 2
    job_dir = 'test'
    if cuda:
        if nr_of_gpus>1:
            devices = GPUtil.getAvailable(order='memory',limit=nr_of_gpus)
            residual = seed_max%nr_of_gpus
            interval_size = (seed_max-residual)/nr_of_gpus
            jobs = []
            for i in range(nr_of_gpus):
                if i == nr_of_gpus-1:
                    jobs.append([int(i*interval_size),int((i+1)*interval_size+residual)])
                else:
                    jobs.append([int(i*interval_size),int((i+1)*interval_size)])

            print(jobs)
            processes = []
            mp.set_start_method('spawn')
            for i in range(nr_of_gpus):
                seed_a = jobs[i][0]
                seed_b = jobs[i][1]
                p = mp.Process(target=run_jobs, args=(seed_a,seed_b,theta,phi,beta_XZ_list,n_list,devices[i],job_dir))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        else:
            device = GPUtil.getFirstAvailable(order='memory')[0]
            seed_a = 0
            seed_b = seed_max
            run_jobs(seed_a=seed_a,seed_b=seed_b,theta=theta,phi=phi,beta_XZ_list=beta_XZ_list,n_list=n_list,device=device,job_dir=job_dir)
            #Figure out what the hell is going on?! why would it work with "X"?!
    else:
        device = "cpu"
        seed_a = 0
        seed_b = seed_max
        run_jobs(seed_a=seed_a, seed_b=seed_b, theta=theta, phi=phi, beta_XZ_list=beta_XZ_list, n_list=n_list,
                 device=device,job_dir=job_dir)
