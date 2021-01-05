import os
import shutil
import pickle

def save_obj(obj, name ,folder):
    with open(f'{folder}'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name,folder):
    with open(f'{folder}' + name, 'rb') as f:
        return pickle.load(f)

def generate_job_params(n_list,net_width,net_layers,runs=1,seed_max=1000,estimate=False,directory='job_dir/'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory)
    counter = 0
    for n in n_list:
        for d_X, d_Y, d_Z, theta, phi in zip([1],[1],[1],[2.0],[2.0]):
                # zip([1,3, 3, 3], [1,3, 3, 3], [1,3, 15, 50], [2.0,3.0, 8.0, 16.0],
                #                              [2.0, 2.0, 2.0, 2.0]):  # 50,3
            for beta_XZ in [0.5]:
                for q in [0.75]:
                    for by in [0.1,0.25,0.5,0.0]: #Robin suggest: [0.0, 0.1,0.25,0.5]
                        h_0_test = f'univariate_{seed_max}_seeds/univariate_test'
                        ba = 0
                        if d_X==1:
                            ba = 0.0
                            if by==0.0:
                                by=0
                        # if d_X==3:
                        #     ba=0
                        beta_xy = [ba, by]
                        data_dir = f"data_100/beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n=10000_yz=0.5_beta_XZ={beta_XZ}_theta={theta}_phi={phi}"
                        # mv_str = f'q=1.0_mv_100/beta_xy=[0, {by}]_d_X=3_d_Y=3_d_Z={d_Z}_n=10000_yz=0.5_beta_XZ={beta_XZ}_theta={theta}_phi={phi}/'
                        # uni_str = f'univariate_100_seeds/Q=1.0_gt=H_{h_int}_y_a=0.0_y_b={by}_z_a=0.0_z_b={beta_XZ}_cor=0.5_n=10000_seeds=100_{theta}_{phi}/'
                        val_rate = 0.1
                        h_str =data_dir
                        if estimate:
                            models_to_run = zip(['real_TRE_Q','NCE_Q'],[1,10])
                            # models_to_run = zip(['rulsif'],[1,1])
                        else:
                            models_to_run = zip(['real_weights'],[1])
                        for mode in ['Q']:
                            for width in net_width:
                                for layers in net_layers:
                                    job_dir = f'layers={layers}_width={width}'
                                    for h in [h_str]:
                                        for model,kappa in models_to_run:#zip(['real_TRE_Q'],[1]):# zip(['TRE_Q','NCE_Q','NCE'],[1,10,10]):
                                            for br in [250]:# zip([h_0_str_mult_2_big,h_1_str_mult_2_big],[seed_max,seed_max]):
                                                args = {
                                                    'device': -1,
                                                    'job_dir':job_dir,
                                                    'data_dir': h,
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
                                                                   'm': 4
                                                                   },
                                                    'estimator': model, #ones, 'NCE'
                                                    'runs': runs,
                                                    'cuda': True,
                                                }
                                                save_obj(args,f'job_{counter}',directory)
                                                counter+=1

if __name__ == '__main__':
    generate_job_params(n_list=[1000,5000,10000],net_layers=[3],net_width=[32],runs=1,seed_max=100,estimate=True,directory='job_univariate/')
    generate_job_params(n_list=[1000,5000,10000],net_layers=[3],net_width=[32],runs=1,seed_max=100,estimate=False,directory='job_univariate_real/')