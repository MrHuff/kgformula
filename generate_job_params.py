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




def gen_hdm_breaker(n_list,net_width,net_layers,runs=1,seed_max=1000,estimate=False,directory='job_dir/',job_type='kc',dirname=''):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory)
    yz = [0.5, 0.0]
    counter = 0
    PHI = [2.0,2.0]
    THETA = [2.0,16.0]
    DX = [1,3]
    DY = [1,3]
    DZ = [1,50]
    for d_X, d_Y, d_Z, theta, phi,beta_XZ in zip(DX,DY,DZ,THETA,PHI,[0.25,0.075]):
        for n in n_list:
            for q in [1.0]:
                if d_X==1:
                    betas = [[0,0.0],[0,0.01],[0,0.02],[0,0.03],[0,0.04],[0,0.05]]
                else:
                    betas = [[0,0.0],[0,1e-4],[0,2e-4],[0,3e-4],[0,4e-4],[0,5e-4]]
                for beta_xy in betas:
                    data_dir = f"{dirname}/beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n=10000_yz={yz}_beta_XZ={beta_XZ}_theta={theta}_phi={phi}"
                    val_rate = 0.2
                    job_character = {
                        'beta_xy': beta_xy,
                        'd_X': d_X,
                        'd_Y': d_Y,
                        'd_Z': d_Z,
                        'n': n,
                        'yz': [0.5, 0],
                        'beta_XZ': beta_XZ,
                        'theta': theta,
                        'phi': phi
                    }
                    h_str =data_dir
                    if estimate:
                        models_to_run = zip(['real_weights','NCE_Q','random_uniform','rulsif'],[1,10,1,1])
                    else:
                        models_to_run = zip(['real_weights'],[1])
                    for mode in ['Q']:
                        for width in net_width:
                            for layers in net_layers:
                                job_dir = f'{directory}'
                                for model,kappa in models_to_run:#zip(['real_TRE_Q'],[1]):# zip(['TRE_Q','NCE_Q','NCE'],[1,10,10]):
                                    for br in [500]:# zip([h_0_str_mult_2_big,h_1_str_mult_2_big],[seed_max,seed_max]):
                                        args = {
                                            'job_character': job_character,
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
                                                           'max_its': 10,
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
                                            'variant':1
                                        }
                                        save_obj(args,f'job_{counter}',directory+'/')
                                        counter+=1


def generate_job_params(n_list,net_width,net_layers,runs=1,seed_max=100,estimate=False,directory='job_dir/',job_type='kc',variant=1,exp_mode=1):
    """
    Base jobs
    """
    if exp_mode==1:
        yz = [0.5, 0.0]
        b_z = [0.25]
        dirname = 'do_null_100'
        PHI = [2.0,2.0,2.0,2.0]
        THETA = [2.0,4.0,16.0,64.0]
        DX = [1,3,3,3]
        DY = [1,3,3,3]
        DZ = [1,3,15,50]

    if exp_mode=='debug_layernorm':
        yz = [0.5, 0.0]
        b_z = [0.25]
        dirname = 'do_null_100'
        PHI = [2.0,2.0]
        THETA = [8.0,16.0]
        DX = [3,3]
        DY = [3,3]
        DZ = [15,50]

    if exp_mode == 'debug':
        yz = [0.5, 0.0]
        b_z = [0.25]
        dirname = 'debug_do_null_100'
        PHI = [ 2.0, 2.0]
        THETA = [32.0, 32.0]
        DX = [3, 3]
        DY = [3, 3]
        DZ = [15, 50]
    if exp_mode == 'debug_2':
        yz = [0.5, 0.0]
        b_z = [0.25]
        dirname = 'debug_2_do_null_100'
        PHI = [ 2.0]
        THETA = [64.0]
        DX = [ 3]
        DY = [3]
        DZ = [50]
    if exp_mode == 'debug_3':
        yz = [0.5, 0.0]
        b_z = [0.25]
        dirname = 'debug_3_do_null_100'
        PHI = [2.0, 2.0]
        THETA = [16.0, 48.0]
        DX = [3, 3]
        DY = [3, 3]
        DZ = [15, 50]
        """
        Break marginal
        """
    if exp_mode == 'debug_4':
        yz = [0.5, 0.0]
        b_z = [0.25]
        dirname = 'debug_4_do_null_100'
        PHI = [2.0, 2.0]
        THETA = [12.0, 72.0]
        DX = [3, 3]
        DY = [3, 3]
        DZ = [15, 50]
        """
        Break marginal
        """
    elif exp_mode==2:
        dirname = 'exp_hsic_break_{N}'
        yz = [0.5, 0.0]
        b_z =  [0.0]
        PHI = [1.5]
        THETA = [0.1]
        BXY_list = [0.0]
        DX = [1]
        DY = [1]
        DZ = [1]

    elif exp_mode==3:
        """
        Break conditional
        """
        dirname = 'exp_gcm_break_{N}'
        yz=[-0.5,4.0]
        b_z =  [0.0]
        PHI = [1.0]
        THETA = [2.0]
        BXY_list = [0.0]
        DX = [1]
        DY = [1]
        DZ = [1]

    Q_LIST = [1.0]
    BR = [500]
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory)
    counter = 0
    mode='Q'
    for d_X, d_Y, d_Z, theta, phi in zip(DX,DY,DZ,THETA,PHI):
        for beta_XZ in b_z:
            for n in n_list:
                for q in Q_LIST:
                    # for beta_xy in [[0, 0.0], [0, 0.001], [0, 0.002], [0, 0.003], [0, 0.004], [0, 0.005], [0, 0.008],
                    #                 [0, 0.012], [0, 0.016], [0, 0.02], [0, 0.03], [0, 0.04], [0, 0.05]]:
                    for beta_xy in [[0, 0.0], [0, 0.001], [0, 0.002], [0, 0.003], [0, 0.004], [0, 0.005], [0, 0.008],
                                    [0, 0.012], [0, 0.016], [0, 0.02]]:
                        data_dir = f"{dirname}/beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n=10000_yz={yz}_beta_XZ={beta_XZ}_theta={theta}_phi={phi}"
                        job_character = {
                            'beta_xy': beta_xy,
                            'd_X': d_X,
                            'd_Y': d_Y,
                            'd_Z': d_Z,
                            'n': n,
                            'yz': [0.5, 0],
                            'beta_XZ': beta_XZ,
                            'theta': theta,
                            'phi': phi
                        }
                        h_str =data_dir
                        if estimate:
                            models_to_run = zip(['real_weights','real_TRE_Q','NCE_Q','random_uniform','rulsif'],[1,1,10,1,1])
                        else:
                            models_to_run = zip(['real_weights'],[1])
                        for width in net_width:
                            for layers in net_layers:
                                job_dir = f'{directory}'
                                for model,kappa in models_to_run:#zip(['real_TRE_Q'],[1]):# zip(['TRE_Q','NCE_Q','NCE'],[1,10,10]):
                                    args = {
                                        'job_character': job_character,
                                        'job_type':job_type,
                                        'device': -1,
                                        'job_dir':job_dir,
                                        'data_dir': h_str,
                                        'estimate': estimate,
                                        'debug_plot': False,
                                        'seeds_a': 0,
                                        'seeds_b': seed_max,
                                        'bootstrap_runs': 500,    #play with this (increase it!)
                                        'mode': mode,
                                        'split': estimate,
                                        'q_factor':q,
                                        'qdist': 2,
                                        'n':n,
                                        'est_params': {'lr': 1e-3,
                                                       # use really small LR for TRE. Ok what the fuck is going on...
                                                       'max_its': 50,
                                                       'width': width,
                                                       'layers': layers,
                                                       'mixed': False,
                                                       'bs_ratio': 1.0,
                                                       'val_rate': 0.2,
                                                       'n_sample': 250,
                                                       'criteria_limit': 0.05,
                                                       'kill_counter': 10,
                                                       'kappa': kappa,
                                                       'separate': False,
                                                       'm': 4
                                                       },
                                        'estimator': model, #ones, 'NCE'
                                        'runs': runs,
                                        'cuda': True,
                                        'sanity_exp': False,
                                        'variant':variant
                                    }
                                    save_obj(args,f'job_{counter}',directory+'/')
                                    counter+=1
                                    print(counter)


def generate_job_kchsic_breaker_linear(n_list,net_width,net_layers,runs=1,seed_max=1000,estimate=False,directory='job_dir/',job_type='kc',dirname='',theta=2.0,phi=2.0,variant=1):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory)
    counter = 0
    DX=[1]
    DY=[1]
    DZ=[1]
    THETA=[theta]
    PHI=[phi]
    yz = [0.5, 0.0]

    for d_X, d_Y, d_Z, theta, phi in zip(DX,DY,DZ,THETA,PHI):
        for beta_XZ in [0.0,0.05,0.1,0.25,0.5,0.75,1.0,1.5,2.0,3.0,5.0]:
            for n in [10000]:
                for q in [1.0]:
                    for by in [0.0]: #Robin suggest: [0.0, 0.1,0.25,0.5]
                        ba = 0
                        beta_xy = [ba, by]
                        data_dir = f"{dirname}/beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n=10000_yz={yz}_beta_XZ={beta_XZ}_theta={theta}_phi={phi}"
                        job_character={
                            'beta_xy':beta_xy,
                            'd_X':d_X,
                            'd_Y':d_Y,
                            'd_Z':d_Z,
                            'n':n,
                            'yz':yz,
                            'beta_XZ':beta_XZ,
                            'theta':theta,
                            'phi':phi
                        }
                        val_rate = 0.2
                        h_str =data_dir
                        if estimate:
                            models_to_run = zip(['real_weights','real_TRE_Q','NCE_Q',],[1,1,10])
                            # models_to_run = zip(['real_weights','NCE_Q'],[1,10])
                            # models_to_run = zip(['real_weights','random_uniform','NCE_Q'],[1,1,10])
                            # models_to_run = zip(['rulsif'],[1,1])
                        else:
                            models_to_run = zip(['real_weights'],[1])
                        for mode in ['Q']:
                            for width in net_width:
                                for layers in net_layers:
                                    job_dir = f'{directory}'
                                    for model,kappa in models_to_run:#zip(['real_TRE_Q'],[1]):# zip(['TRE_Q','NCE_Q','NCE'],[1,10,10]):
                                        for br in [500]:# zip([h_0_str_mult_2_big,h_1_str_mult_2_big],[seed_max,seed_max]):
                                            args = {
                                                'job_character':job_character,
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
                                                'est_params': {'lr': 1e-3, #use really small LR for TRE. Ok what the fuck is going on...
                                                               'max_its': 50,
                                                               'width': width,
                                                               'layers':layers,
                                                               'mixed': False,
                                                               'bs_ratio': 1.0,
                                                               'val_rate': val_rate,
                                                               'n_sample': 250,
                                                               'criteria_limit': 0.05,
                                                               'kill_counter': 10,
                                                                'kappa':kappa,
                                                               'separate':False,
                                                               'm': 3
                                                               },
                                                'estimator': model, #ones, 'NCE'
                                                'runs': runs,
                                                'cuda': True,
                                                'sanity_exp': False,
                                                'variant':variant
                                            }
                                            save_obj(args,f'job_{counter}',directory+'/')
                                            counter+=1

def generate_job_kchsic_breaker(n_list,net_width,net_layers,runs=1,seed_max=1000,estimate=False,directory='job_dir/',job_type='kc',dirname='',theta=4.0,phi=2.0):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory)
    counter = 0
    DX=[3]
    DY=[3]
    DZ=[15]
    THETA=[theta]
    PHI=[phi]
    yz = [0.5, 0.0]

    for d_X, d_Y, d_Z, theta, phi in zip(DX,DY,DZ,THETA,PHI):
        for beta_XZ in [0.0,0.05,0.1,0.25,0.5,0.75,1.0,1.5,2.0,3.0,5.0]:
            for n in [10000]:
                for q in [1.0]:
                    for by in [0.0]: #Robin suggest: [0.0, 0.1,0.25,0.5]
                        ba = 0
                        beta_xy = [ba, by]
                        data_dir = f"{dirname}/beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n=10000_yz={yz}_beta_XZ={beta_XZ}_theta={theta}_phi={phi}"
                        job_character={
                            'beta_xy':beta_xy,
                            'd_X':d_X,
                            'd_Y':d_Y,
                            'd_Z':d_Z,
                            'n':n,
                            'yz':yz,
                            'beta_XZ':beta_XZ,
                            'theta':theta,
                            'phi':phi
                        }
                        val_rate = 0.2
                        h_str =data_dir
                        if estimate:
                            models_to_run = zip(['real_weights','real_TRE_Q','NCE_Q',],[1,1,10])
                            # models_to_run = zip(['real_weights','NCE_Q'],[1,10])
                            # models_to_run = zip(['real_weights','random_uniform','NCE_Q'],[1,1,10])
                            # models_to_run = zip(['rulsif'],[1,1])
                        else:
                            models_to_run = zip(['real_weights'],[1])
                        for mode in ['Q']:
                            for width in net_width:
                                for layers in net_layers:
                                    job_dir = f'{directory}'
                                    for model,kappa in models_to_run:#zip(['real_TRE_Q'],[1]):# zip(['TRE_Q','NCE_Q','NCE'],[1,10,10]):
                                        for br in [500]:# zip([h_0_str_mult_2_big,h_1_str_mult_2_big],[seed_max,seed_max]):
                                            args = {
                                                'job_character':job_character,
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
                                                'est_params': {'lr': 1e-3, #use really small LR for TRE. Ok what the fuck is going on...
                                                               'max_its': 50,
                                                               'width': width,
                                                               'layers':layers,
                                                               'mixed': False,
                                                               'bs_ratio': 1.0,
                                                               'val_rate': val_rate,
                                                               'n_sample': 250,
                                                               'criteria_limit': 0.05,
                                                               'kill_counter': 10,
                                                                'kappa':kappa,
                                                               'separate':False,
                                                               'm': 3
                                                               },
                                                'estimator': model, #ones, 'NCE'
                                                'runs': runs,
                                                'cuda': True,
                                                'sanity_exp': False,
                                                'variant':1
                                            }
                                            save_obj(args,f'job_{counter}',directory+'/')
                                            counter+=1


def generate_job_params_HSIC(n_list,runs=1,seed_max=1000,estimate=False,directory='job_dir/'):
    N=100
    BXY_const = 0.0
    yz=[0.5,0.0]
    b_z = [1e-2,0.05,0.1,0.25,0.5,1.0]
    dirname ='exp_hsic_break_100'
    PHI=[1.5]
    THETA=[0.1]
    DX=[1]
    DY = [1]
    DZ = [1]
    BR = [500]
    BXY_list = [0.0]
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
                    job_character = {
                        'beta_xy': beta_xy,
                        'd_X': d_X,
                        'd_Y': d_Y,
                        'd_Z': d_Z,
                        'n': n,
                        'yz': yz,
                        'beta_XZ': beta_XZ,
                        'theta': theta,
                        'phi': phi
                    }
                    h_str =f'{data_dir}'
                    models_to_run = zip(['real_weights'],[1])
                    job_dir = f'{directory}'
                    for model,kappa in models_to_run:#zip(['real_TRE_Q'],[1]):# zip(['TRE_Q','NCE_Q','NCE'],[1,10,10]):
                        for br in BR:# zip([h_0_str_mult_2_big,h_1_str_mult_2_big],[seed_max,seed_max]):
                            args = {
                                'job_character': job_character,
                                'job_type':'hsic',
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

def generate_job_binary(n_list,net_width,net_layers,runs=1,seed_max=1000,estimate=False,directory='job_dir/',job_type='kc',variant=1):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory)
    counter = 0
    for d in [1]:
        for n in n_list:
            for alp in [0.0,0.005,0.01,2*1e-2,4*1e-2,6*1e-2,8*1e-2,1e-1]:
            # for alp in [4 * 1e-2]:
                for null_case in [False]:
                    for sep in [True]:
                        data_dir = f'do_null_univariate_alp={alp}_null={null_case}_d={d}'
                        val_rate = 0.2
                        h_str =data_dir

                        job_character = {
                            'beta_xy': [0.0,alp],
                            'd_X': 1,
                            'd_Y': 1,
                            'd_Z': 1,
                            'n': n,
                            'yz': 0.5,
                            'beta_XZ': 0.5,
                            'theta': 1.,
                            'phi': 1.
                        }
                        if estimate:
                            # models_to_run = zip(['real_TRE_Q', 'NCE_Q'], [1, 10, ])
                            # models_to_run = zip(['cfme','old_statistic'], [1,1])
                            models_to_run = zip([ 'NCE_Q','real_weights'], [10,1])
                        else:
                            models_to_run = zip(['real_weights'],[1])
                        for mode in ['Q']:
                            for width in net_width:
                                for layers in net_layers:
                                    job_dir = f'{directory}'
                                    for model,kappa in models_to_run:#zip(['real_TRE_Q'],[1]):# zip(['TRE_Q','NCE_Q','NCE'],[1,10,10]):
                                        for br in [500]:# zip([h_0_str_mult_2_big,h_1_str_mult_2_big],[seed_max,seed_max]):
                                            args = {
                                                'job_character': job_character,
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
                                                               'max_its': 50,
                                                               'width': width,
                                                               'layers':layers,
                                                               'mixed': False,
                                                               'bs_ratio': 1,
                                                               'val_rate': val_rate,
                                                               'n_sample': 250,
                                                               'criteria_limit': 0.05,
                                                               'kill_counter': 10,
                                                                'kappa':kappa,
                                                               'm': 4,
                                                               'separate': sep
                                                               },
                                                'estimator': model, #ones, 'NCE'
                                                'runs': runs,
                                                'cuda': True,
                                                'sanity_exp': False,
                                                'variant':variant
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
        sep_list = [ True,False]
    else:
        models,kappas = ['real_weights'], [1]
        sep_list = [False]
    counter = 0
    for d_X, d_Y, d_Z, theta, phi,beta_XZ in zip([2, 6, 8], [2,  6, 8], [2,  15, 50], [2.0, 16.0, 16.0],[2.0,  2.0, 2.0],[0.05,0.05,0.05]):
        for beta_xy in [[0, 0.0], [0, 0.002], [0, 0.004], [0, 0.006], [0, 0.008], [0, 0.01],[0, 0.015],[0, 0.02],[0, 0.025],[0, 0.03],[0, 0.04],[0, 0.05],[0, 0.1]]:
            for n in n_list:
                for mode in ['Q']:
                    for width in net_width:
                        for layers in net_layers:
                            for model,kappa in zip(models,kappas):#zip(['real_TRE_Q'],[1]):# zip(['TRE_Q','NCE_Q','NCE'],[1,10,10]):
                                for sep in sep_list:
                                    for br in [500]:# zip([h_0_str_mult_2_big,h_1_str_mult_2_big],[seed_max,seed_max]):

                                        data_dir = f"{data_source}/beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n=10000_yz=[0.5, 0.0]_beta_XZ={beta_XZ}_theta={theta}_phi={phi}"
                                        job_character = {
                                            'beta_xy': beta_xy,
                                            'd_X': d_X,
                                            'd_Y': d_Y,
                                            'd_Z': d_Z,
                                            'n': n,
                                            'yz': [0.5,0],
                                            'beta_XZ': beta_XZ,
                                            'theta': theta,
                                            'phi': phi
                                        }
                                        val_rate = 0.1
                                        job_dir = f'{directory}'
                                        args = {
                                            'job_character': job_character,
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
                                            'est_params': {'lr': 1e-3, #use really small LR for TRE. Ok what the fuck is going on...
                                                           'max_its': 10,
                                                           'width': width,
                                                           'layers':layers,
                                                           'mixed': False,
                                                           'bs_ratio': 1.0,
                                                           'val_rate': val_rate,
                                                           'n_sample': 500,
                                                           'criteria_limit': 0.05,
                                                           'kill_counter': 10,
                                                            'kappa':kappa,
                                                           'm': 4,
                                                           'separate':sep
                                                           },
                                            'estimator': model, #ones, 'NCE'
                                            'runs': runs,
                                            'cuda': True,
                                            'sanity_exp': False,
                                            'variant':1,
                                        }
                                        save_obj(args,f'job_{counter}',directory+'/')
                                        counter+=1
                                        print(args)

if __name__ == '__main__':
    for fam_y in [1,4]:
        gen_hdm_breaker(n_list=[1000,5000,10000],net_layers=[2],net_width=[32],runs=1,seed_max=100,estimate=True,directory=f'hdm_breaker_fam_y={fam_y}_job_perm_cluster',job_type='kc_rule_correct_perm',dirname=f'hdm_breaker_fam_y={fam_y}_100')
    generate_job_kchsic_breaker(n_list=[10000],net_layers=[2],net_width=[32],runs=1,seed_max=100,estimate=True,directory='bdhsic_breaker_6_correct_clust_perm',job_type='kc_rule_correct_perm',dirname='bdhsic_breaker_2_100',theta=64.0,phi=2.0)
    generate_job_kchsic_breaker_linear(n_list=[10000],net_layers=[2],net_width=[32],runs=1,seed_max=100,estimate=True,directory='bdhsic_breaker_1dlinear_clust_perm',job_type='kc_rule_correct_perm',dirname='bdhsic_breaker_linear_100',theta=2.0,phi=2.0,variant=2)
    generate_job_kchsic_breaker_linear(n_list=[10000],net_layers=[2],net_width=[32],runs=1,seed_max=100,estimate=True,directory='bdhsic_breaker_1d_regular_clust_perm',job_type='kc_rule_correct_perm',dirname='bdhsic_breaker_linear_100',theta=2.0,phi=2.0,variant=1)
    generate_job_params(n_list=[1000,5000,10000],net_layers=[2],net_width=[32],runs=1,seed_max=100,estimate=True,directory='base_jobs_cont_final_cluster',job_type='kc_rule_correct_perm',exp_mode=1)

    # generate_job_params(n_list=[1000,5000,10000],net_layers=[2],net_width=[32],runs=1,seed_max=100,estimate=True,directory='linear',job_type='kc_rule_new',variant=2,exp_mode=1)
    # generate_job_params(n_list=[1000,5000,10000],net_layers=[2],net_width=[32],runs=1,seed_max=100,estimate=True,directory='break_marginal',job_type='kc_rule_new',exp_mode=2)
    # generate_job_params(n_list=[1000,5000,10000],net_layers=[2],net_width=[32],runs=1,seed_max=100,estimate=True,directory='break_conditional',job_type='kc_rule_new',exp_mode=3)

    generate_job_binary(n_list=[1000,5000,10000],net_layers=[1],net_width=[32],runs=1,seed_max=100,estimate=True,directory='do_null_binary_linear_kernel_perm_cluster',job_type='kc_rule_correct_perm',variant=2)
    generate_job_binary(n_list=[1000,5000,10000],net_layers=[1],net_width=[32],runs=1,seed_max=100,estimate=True,directory='do_null_binary_perm_cluster',job_type='kc_rule_correct_perm',variant=1)
    generate_job_mixed(data_source='do_null_mix_new_100',n_list=[1000,5000,10000],net_layers=[2],net_width=[32],runs=1,seed_max=100,estimate=True,directory='do_null_mix_jobs_cluster',job_type='kc_rule_correct_perm')

    """
    BASELINES
    """

    # generate_job_binary(n_list=[1000,5000,10000],net_layers=[1],net_width=[32],runs=1,seed_max=100,estimate=False,directory='do_null_binary_bench_cfme',job_type='cfme',variant=1)
    # generate_job_binary(n_list=[1000,5000,10000],net_layers=[1],net_width=[32],runs=1,seed_max=100,estimate=False,directory='do_null_binary_bench_old_statistic',job_type='old_statistic',variant=1)
    # generate_job_mixed(data_source='do_null_mix_new_100',n_list=[1000,5000,10000],net_layers=[2],net_width=[32],runs=1,seed_max=100,estimate=False,directory='do_null_mix_jobs_old_statistic',job_type='old_statistic')
    # for fam_y in [1,4]:
    #     gen_hdm_breaker(n_list=[1000,5000,10000],net_layers=[2],net_width=[32],runs=1,seed_max=100,estimate=False,directory=f'hdm_breaker_fam_y={fam_y}_job_old_statistic',job_type='old_statistic',dirname=f'hdm_breaker_fam_y={fam_y}_100')

    generate_job_kchsic_breaker(n_list=[10000],net_layers=[2],net_width=[32],runs=1,seed_max=100,estimate=False,directory='bdhsic_breaker_6_correct_old_statistic',job_type='old_statistic',dirname='bdhsic_breaker_2_100',theta=64.0,phi=2.0)
    generate_job_kchsic_breaker_linear(n_list=[10000],net_layers=[2],net_width=[32],runs=1,seed_max=100,estimate=False,directory='bdhsic_breaker_1dlinear_old_statistic',job_type='old_statistic',dirname='bdhsic_breaker_linear_100',theta=2.0,phi=2.0)

    # generate_job_params(n_list=[1000,5000,10000],net_layers=[2],net_width=[32],runs=1,seed_max=100,estimate=False,directory='base_jobs_cont_final_old_statistic',job_type='old_statistic',exp_mode=1)

    # generate_job_params_HSIC(n_list=[1000,5000,10000],directory='hsic_baseline')


    #TODO investigate switching back to "old incorrect test statistic". Could investigate implementation of permutation... - did not too mcuh
    #Could be training or architectural choice... Shouldn't matter that much though...