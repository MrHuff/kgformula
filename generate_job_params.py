import os
import shutil
import pickle
from itertools import *

def save_obj(obj, name ,folder):
    with open(f'{folder}'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name,folder):
    with open(f'{folder}' + name, 'rb') as f:
        return pickle.load(f)


# KC-HSIC BREAK
# N=100
# BXY_const = 0
# BXY = 0.5
# yz=[0.5,0.0]
# b_z =[0.0,0.5]#[0.0,0.25,0.5,0.75,1.0,2.0,5.0,10.0]
# dirname ='kchsic_break_100'
# PHI=[1.0]
# THETA=[0.1,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]
# DX=[1]
# DY = [1]
# DZ = [1]
# BR = [500]
# BXY_list = [0.0]
# Q_LIST=[1.0]
# MAX_ITS=10





#COND BREAK
# N=100
# BXY_const = 0.0
# BXY = 0.5
# yz=[-0.5,4.0]
# # yz=[-0.5,1.0]
# b_z = [0.0]
# dirname ='exp_gcm_break_100'
# PHI=[2.0]
# THETA=[1.0]
# DX=[1]
# DY = [1]
# DZ = [1]
# BXY_list = [0.0]
# # BXY_list = [0.0]
# Q_LIST=[1.0]
# BR = [500]
# MAX_ITS=10




#HSIC BREAK
# N=100
# BXY_const = 0.0
# BXY = 0.5
# yz=[0.5,0.0]
# b_z = [1e-2,0.05,0.1,0.25,0.5,1.0]
# dirname ='exp_hsic_break_100'
# PHI=[1.5]
# THETA=[0.1]
# # PHI=[0.5]
# # THETA=[5.0]
# DX=[1]
# DY = [1]
# DZ = [1]
# BR = [500]
# BXY_list = [0.0]
# # BXY_list = [0.1]
# Q_LIST=[1.0]
# MAX_ITS=10

#Regular setup
N=100
BXY_const = 0
yz=[0.5,0.0]
b_z = [0.75]
dirname ='do_null_100'
# PHI=[2.0,2.0,2.0]
# THETA=[4.0,8.0,16.0]
# PHI=[2.0,2.0]
# THETA=[8.0,16.0]
PHI=[2.0]
THETA=[2.0]
# BXY_list = [0.0,0.05,0.1,0.15,0.2]
BXY_list = [0.0,0.01,0.02,0.03,0.04,0.05]
# BXY_list = [0.0,0.001,0.002,0.003,0.004,0.005]
# BXY_list = [0.0,0.004,0.008,0.012,0.016,0.02]
# DX= [3,3,3]
# DY =  [3,3,3]
# DZ = [3,15,50]

# DX= [3,3]
# DY =  [3,3]
# DZ = [15,50]
DX= [1]
DY =  [1]
DZ = [1]

Q_LIST=[1.0]
BR = [500]
MAX_ITS=10


# N=100
# BXY_const = 0.0
# yz=[0.5,0.0]
# b_z = [0.5]
# dirname ='ablation_100'
# PHI=[2.0]
# THETA=[2.0]
# BXY_list = [0.01,0.03,0.05,0.07,0.09]
# DX= [2]
# DY =  [2]
# DZ = [2]
# Q_LIST=[1.0]
# BR = [500]
# MAX_ITS=10
def generate_job_params(n_list,net_width,net_layers,runs=1,seed_max=1000,estimate=False,directory='job_dir/',job_type='kc',dirname=''):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory)
    counter = 0
    # els = list(product(*[DX, DY, DZ, THETA,PHI]))
    # for d_X, d_Y, d_Z, theta, phi in els:
    # for d_X, d_Y, d_Z, theta, phi,beta_XZ in zip(DX,DY,DZ,THETA,PHI,b_z):
    for d_X, d_Y, d_Z, theta, phi in zip(DX,DY,DZ,THETA,PHI):
        for beta_XZ in b_z:
            for n in n_list:
                for q in Q_LIST:
                    for by in BXY_list: #Robin suggest: [0.0, 0.1,0.25,0.5]
                        ba = BXY_const
                        beta_xy = [ba, by]
                        data_dir = f"{dirname}/beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n=10000_yz={yz}_beta_XZ={beta_XZ}_theta={theta}_phi={phi}"
                        val_rate = 0.2
                        h_str =data_dir
                        if estimate:
                            models_to_run = zip(['real_weights','real_TRE_Q','NCE_Q','random_uniform','rulsif'],[1,1,10,1,1])
                            # models_to_run = zip(['random_uniform','NCE_Q'],[1,10])
                            # models_to_run = zip(['rulsif'],[1,1])
                        else:
                            models_to_run = zip(['real_weights'],[1])
                        for mode in ['Q']:
                            for width in net_width:
                                for layers in net_layers:
                                    job_dir = f'{directory}_layers={layers}_width={width}'
                                    for model,kappa in models_to_run:#zip(['real_TRE_Q'],[1]):# zip(['TRE_Q','NCE_Q','NCE'],[1,10,10]):
                                        for br in BR:# zip([h_0_str_mult_2_big,h_1_str_mult_2_big],[seed_max,seed_max]):
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
                                                'est_params': {'lr': 1e-4, #use really small LR for TRE. Ok what the fuck is going on...
                                                               'max_its': MAX_ITS,
                                                               'width': width,
                                                               'layers':layers,
                                                               'mixed': False,
                                                               'bs_ratio': 0.01,
                                                               'val_rate': val_rate,
                                                               'n_sample': 250,
                                                               'criteria_limit': 0.05,
                                                               'kill_counter': 2,
                                                                'kappa':kappa,
                                                               'separate':False,
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
                for by in BXY_list: #Robin suggest: [0.0, 0.1,0.25,0.5]
                    ba = BXY_const
                    beta_xy = [ba, by]
                    data_dir = f"{dirname}/beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n=10000_yz={yz}_beta_XZ={beta_XZ}_theta={theta}_phi={phi}"
                    h_str =f'{data_dir}'
                    models_to_run = zip(['real_weights'],[1])
                    job_dir = f'{directory}'
                    for model,kappa in models_to_run:#zip(['real_TRE_Q'],[1]):# zip(['TRE_Q','NCE_Q','NCE'],[1,10,10]):
                        for br in BR:# zip([h_0_str_mult_2_big,h_1_str_mult_2_big],[seed_max,seed_max]):
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
        for d_X, d_Y, d_Z, theta, phi in zip(DX, DY, DZ, THETA,PHI):
            for beta_XZ in b_z:
                for by in BXY_list: #Robin suggest: [0.0, 0.1,0.25,0.5]
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


def generate_job_binary(n_list,net_width,net_layers,runs=1,seed_max=1000,estimate=False,directory='job_dir/',job_type='kc',dirname=''):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory)
    counter = 0
    for d in [1]:
        for n in n_list:
            for alp in [0.0,2*1e-2,4*1e-2,6*1e-2,8*1e-2,1e-1]:
            # for alp in [4 * 1e-2]:
                for null_case in [False]:
                    for sep in [True]:
                        data_dir = f'do_null_univariate_alp={alp}_null={null_case}_d={d}'
                        val_rate = 0.2
                        h_str =data_dir
                        if estimate:
                            # models_to_run = zip(['real_TRE_Q', 'NCE_Q'], [1, 10, ])
                            models_to_run = zip([ 'NCE_Q','real_weights'], [10,1])
                        else:
                            models_to_run = zip(['real_weights'],[1])
                        for mode in ['Q']:
                            for width in net_width:
                                for layers in net_layers:
                                    job_dir = f'{directory}_layers={layers}_width={width}_{sep}'
                                    for model,kappa in models_to_run:#zip(['real_TRE_Q'],[1]):# zip(['TRE_Q','NCE_Q','NCE'],[1,10,10]):
                                        for br in BR:# zip([h_0_str_mult_2_big,h_1_str_mult_2_big],[seed_max,seed_max]):
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
                                                'q_factor':1.0,
                                                'qdist': 2,
                                                'n':n,
                                                'est_params': {'lr': 1e-3, #use really small LR for TRE. Ok what the fuck is going on...
                                                               'max_its': MAX_ITS,
                                                               'width': width,
                                                               'layers':layers,
                                                               'mixed': False,
                                                               'bs_ratio': 1e-2,
                                                               'val_rate': val_rate,
                                                               'n_sample': 250,
                                                               'criteria_limit': 0.05,
                                                               'kill_counter': 2,
                                                                'kappa':kappa,
                                                               'm': 4,
                                                               'separate': sep
                                                               },
                                                'estimator': model, #ones, 'NCE'
                                                'runs': runs,
                                                'cuda': True,
                                                'sanity_exp': False,
                                            }
                                            save_obj(args,f'job_{counter}',directory+'/')
                                            counter+=1

def generate_job_mixed(data_source,n_list,net_width,net_layers,runs=1,seed_max=1000,estimate=False,directory='job_dir/',job_type='kc'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory)
    if estimate:
        models,kappas = ['real_weights','real_TRE_Q', 'NCE_Q', 'random_uniform', 'rulsif'],[1,1, 10, 1, 1]
        # models,kappas = [ 'NCE_Q', 'random_uniform'],[10, 1]
        # models,kappas = ['real_weights'],[1]
        sep_list = [ True,False]
        # sep_list = [False]
    else:
        models,kappas = ['real_weights'], [1]
        sep_list = [False]
    counter = 0


    # theta_vec = [16.0]
    # phi_vec = [2.0]
    # d_X=[8]
    # d_Y=[8]
    # d_Z=[50]
    # bz_list=[0.0,0.05,0.1,0.15,0.20,0.25]
    # els = list(product(*[d_X, d_Y, d_Z, theta_vec,phi_vec,bz_list]))
    # beta_XZ_list = [0.0,0.05,0.1,0.25]
    for d_X, d_Y, d_Z, theta, phi,beta_XZ in zip([2, 6, 8], [2,  6, 8], [2,  15, 50], [2.0, 16.0, 16.0],[2.0,  2.0, 2.0],[0.05,0.05,0.05]):
    # for d_X, d_Y, d_Z, theta, phi in zip([2, 6, 8], [2,  6, 8], [2,  15, 50], [2.0, 16.0, 16.0],[2.0,  2.0, 2.0]):
        for beta_xy in [[0, 0.0], [0, 0.002], [0, 0.004], [0, 0.006], [0, 0.008], [0, 0.01],[0, 0.015],[0, 0.02],[0,0.025],[0,0.03],[0,0.04],[0,0.05],[0,0.1]]:
            for n in n_list:
                for mode in ['Q']:
                    for width in net_width:
                        for layers in net_layers:
                            for model,kappa in zip(models,kappas):#zip(['real_TRE_Q'],[1]):# zip(['TRE_Q','NCE_Q','NCE'],[1,10,10]):
                                for sep in sep_list:
                                    for br in BR:# zip([h_0_str_mult_2_big,h_1_str_mult_2_big],[seed_max,seed_max]):
                                        data_dir = f"{data_source}/beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n=10000_yz=[0.5, 0.0]_beta_XZ={beta_XZ}_theta={theta}_phi={phi}"
                                        val_rate = 0.2
                                        job_dir = f'{directory}_layers={layers}_width={width}_sep={sep}'

                                        args = {
                                            'job_type':job_type,
                                            'device': -1,
                                            'job_dir':job_dir,
                                            'data_dir': data_dir,
                                            'estimate': estimate,
                                            'debug_plot': False,
                                            'seeds_a': 0,
                                            'seeds_b': seed_max,
                                            'bootstrap_runs': br, #play with this (increase it!)
                                            'mode': mode,
                                            'split': estimate,
                                            'q_factor':1.0,
                                            'qdist': 2,
                                            'n':n,
                                            'est_params': {'lr': 1e-4, #use really small LR for TRE. Ok what the fuck is going on...
                                                           'max_its': MAX_ITS,
                                                           'width': width,
                                                           'layers':layers,
                                                           'mixed': False,
                                                           'bs_ratio': 1e-2,
                                                           'val_rate': val_rate,
                                                           'n_sample': 250,
                                                           'criteria_limit': 0.05,
                                                           'kill_counter': 2,
                                                            'kappa':kappa,
                                                           'm': 4,
                                                           'separate':sep
                                                           },
                                            'estimator': model, #ones, 'NCE'
                                            'runs': runs,
                                            'cuda': True,
                                            'sanity_exp': False,
                                        }
                                        save_obj(args,f'job_{counter}',directory+'/')
                                        counter+=1
                                        print(args)

if __name__ == '__main__':
    pass
    #Use mixed data to illustrate breaking point!
    #Probably mention that exponential data can blow up at times as well, at least when estimating


    # generate_job_params(n_list=[1000,5000,10000],net_layers=[2],net_width=[32],runs=1,seed_max=100,estimate=True,directory='ablation_mixed_2',job_type='kc_rule_new',dirname=dirname)

    # generate_job_params(n_list=[1000,5000,10000],net_layers=[2],net_width=[32],runs=1,seed_max=100,estimate=True,directory='kc_rule_3_test_2',job_type='kc_rule_new',dirname=dirname)
    # generate_job_params(n_list=[1000,5000,10000],net_layers=[3],net_width=[32],runs=1,seed_max=100,estimate=True,directory='kc_rule_3_test_3d',job_type='kc_rule_new',dirname=dirname)
    # generate_job_params(n_list=[1000,5000,10000],net_layers=[3],net_width=[32],runs=1,seed_max=100,estimate=False,directory='kc_rule_real_weights_3_test_2',job_type='kc_rule_new',dirname=dirname)
    # generate_job_params(n_list=[1000,5000,10000],net_layers=[3],net_width=[32],runs=1,seed_max=100,estimate=True,directory='kc_rule',job_type='kc_rule_new',dirname=dirname)

    # generate_job_binary(n_list=[1000,5000,10000],net_layers=[1],net_width=[32],runs=1,seed_max=100,estimate=True,directory='do_null_binary_all_1d',job_type='kc_rule_new',dirname='')
    generate_job_mixed(data_source='do_null_mix_new_100',n_list=[1000,5000,10000],net_layers=[3],net_width=[32],runs=1,seed_max=100,estimate=True,directory='do_null_mix_sanity_3_est',job_type='kc_rule_new')
    # generate_job_mixed(data_source='do_null_mix_new_100',n_list=[1000,5000,10000],net_layers=[2],net_width=[32],runs=1,seed_max=100,estimate=False,directory='do_null_mixed_real_3d',job_type='kc_rule_new')

    # generate_job_params(n_list=[1000,5000,10000],net_layers=[3],net_width=[32],runs=1,seed_max=100,estimate=True,directory='break_kchsic_jobs_kc_est_debug',job_type='kc_rule_new',dirname=dirname)
    # generate_job_params_HSIC(n_list=[1000,5000,10000],seed_max=N,directory='ind_jobs_hsic_2',dirname=dirname)

    # generate_job_params(n_list=[1000,5000,10000],net_layers=[3],net_width=[32],runs=1,seed_max=100,estimate=True,directory='cond_jobs_kc_est_rule',job_type='kc_rule_new',dirname=dirname)
    # generate_job_params(n_list=[1000,5000,10000],net_layers=[3],net_width=[32],runs=1,seed_max=100,estimate=False,directory='cond_jobs_kc_real_rule',job_type='kc_rule_new',dirname=dirname)
    # generate_job_params_GCM(n_list=[5000],seed_max=N,directory='cond_jobs_regression',dirname=dirname,job_type='regression')
    # generate_job_params(n_list=[1000,5000,10000],net_layers=[3],net_width=[32],runs=1,seed_max=100,estimate=True,directory='break_kchsic_jobs_kc_est',job_type='kc_rule_new',dirname=dirname)
    # generate_job_params(n_list=[1000,5000,10000],net_layers=[3],net_width=[32],runs=1,seed_max=100,estimate=False,directory='hsic_break_real',job_type='kc_rule_new',dirname=dirname)
    # generate_job_params(n_list=[1000,5000,10000],net_layers=[3],net_width=[32],runs=1,seed_max=100,estimate=True,directory='hsic_break_est',job_type='kc_rule_new',dirname=dirname)

