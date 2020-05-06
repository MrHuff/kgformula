from kgformula.utils import simulation_object
if __name__ == '__main__':
    h_0_str = 'ground_truth=H_0_y_a=0.0_y_b=0.0_z_a=0.0_z_b=0.5_cor=0.5_n=1000_seeds=1000'
    h_1_str = 'ground_truth=H_1_y_a=0.0_y_b=0.5_z_a=0.0_z_b=0.5_cor=0.5_n=1000_seeds=1000'
    h_0_str_mult = 'beta_xy=0_d_X=3_d_Y=2_d_Z=5_n=1000_beta_XZ=0.25'
    h_1_str_mult = 'beta_xy=0.5_d_X=3_d_Y=2_d_Z=5_n=1000_beta_XZ=0.25'
    h_0_str_mult_2 = 'beta_xy=[0, 0.0]_d_X=3_d_Y=3_d_Z=3_n=1000_yz=0.5_beta_XZ=[0.0, 0.25, 0.25, 0.25]'
    h_1_str_mult_2 = 'beta_xy=[0, 0.5]_d_X=3_d_Y=3_d_Z=3_n=1000_yz=0.5_beta_XZ=[0.0, 0.25, 0.25, 0.25]'

    args={
        'data_dir': h_1_str_mult_2,
        'estimate':False,
        'debug_plot':False,
        'seeds':100,
        'bootstrap_runs':250,
        'debug_generative_process':False,
        'debug_d_Z':3,
        'est_params' : {'lr': 1e-3,
                  'max_its': 5000,
                  'width': 64,
                  'layers': 4,
                  'mixed': False,
                  'bs_ratio': 1e-2,
                  'kappa': 10,
                  'val_rate':0.01,
                  'n_sample':250,
                  'criteria_limit':0.1,
                  'kill_counter': 10,
                        'reg_lambda':1e-2,
                        'alpha':0.5},

        'estimator':'classifier',
        'runs':1,
        'cuda':True,
    }
    j = simulation_object(args)
    j.run()
    #Figure out what the hell is going on?! why would it work with "X"?!