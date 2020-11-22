from kgformula.utils import simulation_object
import GPUtil
import torch.multiprocessing as mp
import math
import os

def run_jobs(seed_a,seed_b,beta_XZ_list,n_list,device,net_width,net_layers,runs=1,seed_max=1000):
    for beta_XZ in beta_XZ_list:
        for d_X,d_Y,d_Z,theta,phi in zip([1,3,3],[1,3,3],[1, 3, 50], [2.0, 2.0, 8.0],[2.0, 2.0, 2.0]):
            for q in [0.5]:
                for by in [1e-3,1e-2,1e-1,0.1,0.25,0.5]:
                    for i,n in enumerate(n_list):
                        h_int = int(not by == 1)
                        h_0_test = f'univariate_{seed_max}_seeds/univariate_test'
                        beta_xy = [0.0, by]
                        data_dir = f"data_5/beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n={n}_yz=0.5_beta_XZ={beta_XZ}_theta={theta}_phi={phi}"
                        # mv_str = f'q=1.0_mv_100/beta_xy=[0, {by}]_d_X=3_d_Y=3_d_Z={d_Z}_n=10000_yz=0.5_beta_XZ={beta_XZ}_theta={theta}_phi={phi}/'
                        # uni_str = f'univariate_100_seeds/Q=1.0_gt=H_{h_int}_y_a=0.0_y_b={by}_z_a=0.0_z_b={beta_XZ}_cor=0.5_n=10000_seeds=100_{theta}_{phi}/'
                        val_rate = max(1e-2, 10. / n)
                        estimate = True
                        h_str =data_dir
                        for perm in ['Y']:
                            for mode in ['Q']:
                                for width in net_width:
                                    for layers in net_layers:
                                        job_dir = f'layers={layers}_width={width}'
                                        for h in [h_str]:
                                            for variant in [1]:
                                                for model,kappa in  zip(['real_TRE_Q','TRE_Q','NCE_Q'],[1,1,10]):#zip(['real_TRE_Q'],[1]):# zip(['TRE_Q','NCE_Q','NCE'],[1,10,10]):
                                                    for br in [250]:# zip([h_0_str_mult_2_big,h_1_str_mult_2_big],[seed_max,seed_max]):
                                                        args = {
                                                            'device': device,
                                                            'job_dir':job_dir,
                                                            'data_dir': h,
                                                            'estimate': estimate,
                                                            'debug_plot': False,
                                                            'seeds_a': seed_a,
                                                            'seeds_b': seed_b,
                                                            'bootstrap_runs': br, #play with this (increase it!)
                                                            'mode': mode,
                                                            'split': estimate,
                                                            'perm': perm,
                                                            'variant': variant,
                                                            'q_factor':q,
                                                            'qdist': 1,
                                                            'est_params': {'lr': 1e-6, #use really small LR for TRE
                                                                           'max_its': 5000,
                                                                           'width': width,
                                                                           'layers':layers,
                                                                           'mixed': False,
                                                                           'bs_ratio': 0.05,
                                                                           'val_rate': val_rate,
                                                                           'n_sample': 250,
                                                                           'criteria_limit': 0.05,
                                                                           'kill_counter': 10,
                                                                            'kappa':kappa,
                                                                           'm': 3
                                                                           },
                                                            'estimator': model, #ones, 'NCE'
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
    # In real life calculate empirical variance X|Z, regress X and Z and find variance of estimate. Then pick q(X) with smaller variance. q normal or MV.
    # We can use model of q(X) does not have tractable density, closed form is cool but get q(X) from really funky sampler (crazy stuff).
    # wed 26th Aug 2pm UK time

    beta_XZ_list = [0.25, 0.5, 0.01, 0.1, 0.0]
    #0.0,0.001,0.011,0.111
    #0,0.01,0.1,0.25,0.5
    #0.0,0.004,0.02
    n_list = [10000]
    seed_max = 5
    cuda = True
    nr_of_gpus = 1 #Try running single machine for comparison...
    net_layers = [2]#['T']#[2,4] #[2,3] # #[1,2,3] #[1,2,3]
    net_width = [32]#['T']#[128,256,1024,2048] #[32,512] # "['T']#[32,128,512] #[8,16,32]
    if cuda:
        if nr_of_gpus>1:
            devices = GPUtil.getAvailable(order='load',limit=nr_of_gpus)
            print(devices)
            # if 5 in devices:
            #     devices.remove(5)
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
                p = mp.Process(target=run_jobs, args=(seed_a,seed_b,beta_XZ_list,n_list,devices[i],net_width,net_layers,1,seed_max))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        else:
            device = GPUtil.getFirstAvailable(order='memory')[0]
            seed_a = 0
            seed_b = seed_max
            run_jobs(
                     seed_a=seed_a,
                     seed_b=seed_b,
                     beta_XZ_list=beta_XZ_list,
                     n_list=n_list,
                     device=device,
                     net_width=net_width,
                     net_layers=net_layers,
                     seed_max=seed_max
                     )
            #Figure out what the hell is going on?! why would it work with "X"?!
    else:
        device = "cpu"
        seed_a = 0
        seed_b = seed_max
        run_jobs(seed_a=seed_a, seed_b=seed_b, beta_XZ_list=beta_XZ_list, n_list=n_list,
                 device=device,net_width=net_width,net_layers=net_layers,seed_max=seed_max)
