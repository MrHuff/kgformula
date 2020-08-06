from kgformula.utils import simulation_object
import GPUtil
import torch.multiprocessing as mp
import math
import os

def run_jobs(seed_a,seed_b,theta,phi,beta_XZ_list,n_list,device,net_width,net_layers,runs=1,seed_max=1000,scales=[1.0]):
    for beta_XZ in beta_XZ_list:
        for i,n in enumerate(n_list):
            h_0_str = f'univariate_{seed_max}_seeds/ground_truth=H_0_y_a=0.0_y_b=0.0_z_a=0.0_z_b={beta_XZ}_cor=0.5_n={n}_seeds={seed_max}_{theta}_{phi}'
            h_1_str = f'univariate_{seed_max}_seeds/ground_truth=H_1_y_a=0.0_y_b=0.5_z_a=0.0_z_b={beta_XZ}_cor=0.5_n={n}_seeds={seed_max}_{theta}_{phi}'
            h_0_str_mult_2 = f'multivariate_{seed_max}/beta_xy=[0, 0]_d_X=3_d_Y=3_d_Z=3_n={n}_yz=0.5_beta_XZ={beta_XZ}_theta={theta}_phi={phi}'
            h_1_str_mult_2 = f'multivariate_{seed_max}/beta_xy=[0, 1.0]_d_X=3_d_Y=3_d_Z=3_n={n}_yz=0.5_beta_XZ={beta_XZ}_theta={theta}_phi={phi}'
            h_0_str_mult_2_big = f'multivariate_{seed_max}/beta_xy=[0, 0]_d_X=3_d_Y=3_d_Z=50_n={n}_yz=0.5_beta_XZ={beta_XZ}_theta={theta}_phi={phi}'
            h_1_str_mult_2_big = f'multivariate_{seed_max}/beta_xy=[0, 1.0]_d_X=3_d_Y=3_d_Z=50_n={n}_yz=0.5_beta_XZ={beta_XZ}_theta={theta}_phi={phi}'
            val_rate = max(1e-2, 10. / n)
            for s in scales:
                for width in net_width:
                    for layers in net_layers:
                        # layers=int(math.log10(n)-1)
                        # width = 32*2**i
                        job_dir = f'layers={layers}_width={width}_scale={s}'
                        for h in [h_0_str_mult_2]:  # zip([h_0_str_mult_2_big,h_1_str_mult_2_big],[seed_max,seed_max]):
                            args = {
                                'new_consistent':False,
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
                                'est_params': {'lr': 1e-4,
                                               'max_its': 5000,
                                               'width': width,
                                               'mixed': False,
                                               'bs_ratio': 10. / n,
                                               'val_rate': val_rate,
                                               'n_sample': 250,
                                               'criteria_limit': 0.05,
                                               'kill_counter': 10,
                                                 'outputs': [1, 1],
                                              'reg_lambda':1e-1,
                                              'm':n,
                                              'd_X':1,
                                              'd_Z':1,
                                              'latent_dim':16,
                                              'depth_u': 2,
                                              'depth_v': 2,
                                              'IP':False,
                                              'scale_x':s
                                               },
                                'estimator': 'TRE', #ones, 'NCE'
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

    beta_XZ_list = [0.5]#0.5,0.01
    n_list = [10000]
    theta = 12
    phi = round(theta/1.75,2)
    seed_max = 1000
    cuda = True
    nr_of_gpus = 8
    net_layers = [3]#['T']#[2,4] #[2,3] # #[1,2,3] #[1,2,3]
    net_width = [128,512,2048]#['T']#[128,256,1024,2048] #[32,512] # "['T']#[32,128,512] #[8,16,32]
    scales = [1.0,0.75,0.5]
    if cuda:
        if nr_of_gpus>1:
            devices = GPUtil.getAvailable(order='memory',limit=nr_of_gpus)
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
                p = mp.Process(target=run_jobs, args=(seed_a,seed_b,theta,phi,beta_XZ_list,n_list,devices[i],net_width,net_layers,1,seed_max))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        else:
            device = GPUtil.getFirstAvailable(order='memory')[0]
            seed_a = 0
            seed_b = seed_max
            run_jobs(seed_a=seed_a,
                     seed_b=seed_b,
                     theta=theta,
                     phi=phi,
                     beta_XZ_list=beta_XZ_list,
                     n_list=n_list,
                     device=device,
                     net_width=net_width,
                     net_layers=net_layers,
                     seed_max=seed_max,
                     scales=scales)
            #Figure out what the hell is going on?! why would it work with "X"?!
    else:
        device = "cpu"
        seed_a = 0
        seed_b = seed_max
        run_jobs(seed_a=seed_a, seed_b=seed_b, theta=theta, phi=phi, beta_XZ_list=beta_XZ_list, n_list=n_list,
                 device=device,net_width=net_width,net_layers=net_layers,seed_max=seed_max)
