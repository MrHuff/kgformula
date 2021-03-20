import os
import shutil
import pickle

def save_obj(obj, name ,folder):
    with open(f'{folder}'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name,folder):
    with open(f'{folder}' + name, 'rb') as f:
        return pickle.load(f)
# N=100
# BXY_const = 0.0
# BXY = 0.5
# yz=[-0.5,4.0]
# yz=[-0.5,1.0]
# b_z = [0.0]
# dirname ='exp_gcm_break_100'
# PHI=[2.0]
# THETA=[1.0]
# DX=[1]
# DY = [1]
# DZ = [1]
#
N=100
BXY_const = 0.0
BXY = 0.5
yz=[0.5,0.0]
b_z = [0.01]
# b_z = [0.5]
dirname ='exp_hsic_break_100'
# PHI=[0.9] #breaker
# THETA=[0.1] #breaker
PHI=[0.5]
THETA=[5.0]

DX=[1]
DY = [1]
DZ = [1]


def generate_job_params(n_list,net_width,net_layers,runs=1,seed_max=1000,estimate=False,directory='job_dir/',job_type='kc',dirname=''):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory)
    counter = 0
    for n in n_list:
        # for d_X, d_Y, d_Z, theta, phi in zip([1,3, 3, 3], [1,3, 3, 3], [1,3, 15, 50], [1.,1.,1.,1.],[1.,1.,1.,1.]):
        for d_X, d_Y, d_Z, theta, phi in zip(DX, DY, DZ, THETA,PHI):
                # zip([1,3, 3, 3], [1,3, 3, 3], [1,3, 15, 50], [2.0,3.0, 8.0, 16.0],[2.0, 2.0, 2.0, 2.0]):  # 50,3
                # zip([1,3, 3, 3], [1,3, 3, 3], [1,3, 15, 50], [2.0,3.0, 8.0, 16.0],
                #                              [2.0, 2.0, 2.0, 2.0]):  # 50,3
            for beta_XZ in b_z:
                # for q in [1e-2,0.05,0.1,0.25]:
                for q in [0.25]:
                    # for by in [0.0,BXY]: #Robin suggest: [0.0, 0.1,0.25,0.5]
                    for by in [0.5]: #Robin suggest: [0.0, 0.1,0.25,0.5]
                        ba = BXY_const
                        beta_xy = [ba, by]
                        data_dir = f"{dirname}/beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n=10000_yz={yz}_beta_XZ={beta_XZ}_theta={theta}_phi={phi}"
                        val_rate = 0.1
                        h_str =data_dir
                        if estimate:
                            models_to_run = zip(['real_TRE_Q','NCE_Q'],[1,10])
                            # models_to_run = zip(['real_TRE_Q'],[1])
                            # models_to_run = zip(['NCE_Q'],[10])
                            # models_to_run = zip(['rulsif'],[1,1])
                        else:
                            models_to_run = zip(['real_weights'],[1])
                        for mode in ['Q']:
                            for width in net_width:
                                for layers in net_layers:
                                    job_dir = f'{directory}_layers={layers}_width={width}'
                                    for model,kappa in models_to_run:#zip(['real_TRE_Q'],[1]):# zip(['TRE_Q','NCE_Q','NCE'],[1,10,10]):
                                        for br in [250]:# zip([h_0_str_mult_2_big,h_1_str_mult_2_big],[seed_max,seed_max]):
                                            args = {
                                                'job_type':job_type,
                                                'device': -1,
                                                'job_dir':job_dir,
                                                'data_dir': h_str,
                                                'estimate': estimate,
                                                'debug_plot': False,
                                                'seeds_a': 0,
                                                'seeds_b': seed_max,
                                                'bootstrap_runs': br, #play with this (increase it!)
                                                'mode': mode,
                                                'split': estimate,
                                                'q_factor':q,
                                                'qdist': 2,
                                                'n':n,
                                                'est_params': {'lr': 1e-5, #use really small LR for TRE. Ok what the fuck is going on...
                                                               'max_its': 0,
                                                               'width': width,
                                                               'layers':layers,
                                                               'mixed': False,
                                                               'bs_ratio': 0.01,
                                                               'val_rate': val_rate,
                                                               'n_sample': 250,
                                                               'criteria_limit': 0.05,
                                                               'kill_counter': 2,
                                                                'kappa':kappa,
                                                               'm': 4
                                                               },
                                                'estimator': model, #ones, 'NCE'
                                                'runs': runs,
                                                'cuda': True,
                                                'sanity_exp': False,
                                            }
                                            save_obj(args,f'job_{counter}',directory+'/')
                                            counter+=1

def generate_job_params_HSIC(n_list,runs=1,seed_max=1000,estimate=False,directory='job_dir/',job_type='hsic',dirname=''):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory)
    counter = 0
    for n in n_list:
        # for d_X, d_Y, d_Z, theta, phi in zip([1,3, 3, 3], [1,3, 3, 3], [1,3, 15, 50], [1.,1.,1.,1.],[1.,1.,1.,1.]):
        for d_X, d_Y, d_Z, theta, phi in zip(DX, DY, DZ, THETA,PHI):
                # zip([1,3, 3, 3], [1,3, 3, 3], [1,3, 15, 50], [2.0,3.0, 8.0, 16.0],[2.0, 2.0, 2.0, 2.0]):  # 50,3
                # zip([1,3, 3, 3], [1,3, 3, 3], [1,3, 15, 50], [2.0,3.0, 8.0, 16.0],
                #                              [2.0, 2.0, 2.0, 2.0]):  # 50,3
            for beta_XZ in b_z:
                # for q in [1e-2,0.05,0.1,0.25]:
                for by in [0.0,BXY]: #Robin suggest: [0.0, 0.1,0.25,0.5]
                    ba = BXY_const
                    beta_xy = [ba, by]
                    data_dir = f"{dirname}/beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n=10000_yz={yz}_beta_XZ={beta_XZ}_theta={theta}_phi={phi}"
                    h_str =f'{data_dir}'
                    models_to_run = zip(['real_weights'],[1])
                    job_dir = f'{directory}'
                    for model,kappa in models_to_run:#zip(['real_TRE_Q'],[1]):# zip(['TRE_Q','NCE_Q','NCE'],[1,10,10]):
                        for br in [250]:# zip([h_0_str_mult_2_big,h_1_str_mult_2_big],[seed_max,seed_max]):
                            args = {
                                'job_type':job_type,
                                'device': -1,
                                'job_dir':job_dir,
                                'data_dir': h_str,
                                'debug_plot': False,
                                'seeds_a': 0,
                                'seeds_b': seed_max,
                                'bootstrap_runs': br, #play with this (increase it!)
                                'split': estimate,
                                'qdist': 2,
                                'n':n,
                                'estimator': model, #ones, 'NCE'
                                'runs': runs,
                                'cuda': True,
                                'sanity_exp': False,
                            }
                            save_obj(args,f'job_{counter}',directory+'/')
                            counter+=1

def generate_job_params_GCM(n_list,seed_max=1000,directory='job_dir/',job_type='gcm',dirname=''):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory)
    counter = 0
    for n in n_list:
        # for d_X, d_Y, d_Z, theta, phi in zip([1,3, 3, 3], [1,3, 3, 3], [1,3, 15, 50], [1.,1.,1.,1.],[1.,1.,1.,1.]):
        for d_X, d_Y, d_Z, theta, phi in zip(DX, DY, DZ, THETA,PHI):
                # zip([1,3, 3, 3], [1,3, 3, 3], [1,3, 15, 50], [2.0,3.0, 8.0, 16.0],[2.0, 2.0, 2.0, 2.0]):  # 50,3
                # zip([1,3, 3, 3], [1,3, 3, 3], [1,3, 15, 50], [2.0,3.0, 8.0, 16.0],
                #                              [2.0, 2.0, 2.0, 2.0]):  # 50,3
            for beta_XZ in b_z:
                # for q in [1e-2,0.05,0.1,0.25]:
                for by in [0.0,BXY]: #Robin suggest: [0.0, 0.1,0.25,0.5]
                    ba = BXY_const
                    beta_xy = [ba, by]
                    data_dir = f"{dirname}/beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n=10000_yz={yz}_beta_XZ={beta_XZ}_theta={theta}_phi={phi}"
                    h_str =data_dir
                    models_to_run = zip(['real_weights'],[1])
                    job_dir = f'{directory}'
                    for model,kappa in models_to_run:#zip(['real_TRE_Q'],[1]):# zip(['TRE_Q','NCE_Q','NCE'],[1,10,10]):
                            args = {
                                'job_type':job_type,
                                'device': -1,
                                'job_dir':job_dir,
                                'data_dir': h_str,
                                'debug_plot': False,
                                'seeds_a': 0,
                                'seeds_b': seed_max,
                                'n':n,
                                'estimator': model, #ones, 'NCE'
                                'sanity_exp': False,
                            }
                            save_obj(args,f'job_{counter}',directory+'/')
                            counter+=1


if __name__ == '__main__':
    # generate_job_params(n_list=[10000,5000,1000],net_layers=[3],net_width=[32],runs=1,seed_max=100,estimate=False,directory='exp_jobs_true_weights_sanity',exp=True)
    # generate_job_params(n_list=[1000],net_layers=[3],net_width=[32],runs=1,seed_max=100,estimate=False,directory='exp_jobs_hsic',exp=True,job_type='kc')
    # generate_job_params(n_list=[10000,5000,1000],net_layers=[3],net_width=[32],runs=1,seed_max=100,estimate=True,directory='exp_jobs_estimate',exp=True)
    # generate_job_params(n_list=[1000,5000,10000],net_layers=[3],net_width=[32],runs=1,seed_max=100,estimate=False,directory='job_univariate_real/')
    # generate_job_params(n_list=[1000],net_layers=[3],net_width=[32],runs=1,seed_max=100,estimate=False,directory='exp_jobs_test_2',exp=True,job_type='kc')
    generate_job_params(n_list=[1000,5000,10000],net_layers=[3],net_width=[32],runs=1,seed_max=100,estimate=True,directory='exp_jobs_kc_est',job_type='kc',dirname=dirname)
    generate_job_params(n_list=[1000,5000,10000],net_layers=[3],net_width=[32],runs=1,seed_max=100,estimate=False,directory='exp_jobs_kc',job_type='kc',dirname=dirname)
    generate_job_params_GCM(n_list=[1000,5000,10000],seed_max=N,directory='exp_jobs_gcm',dirname=dirname)
    generate_job_params_GCM(n_list=[1000,5000,10000],seed_max=N,directory='exp_jobs_regression',dirname=dirname,job_type='regression')
    generate_job_params_HSIC(n_list=[1000,5000,10000],seed_max=N,directory='exp_jobs_hsic',dirname=dirname)
    # generate_job_params(n_list=[1000],net_layers=[3],net_width=[32],runs=1,seed_max=N,estimate=False,directory='exp_jobs_kc',dirname=dirname)
    # generate_job_params(n_list=[1000],net_layers=[1],net_width=[64],runs=1,seed_max=N,estimate=True,directory='debug_gcm_NCE_Q',dirname=dirname)
