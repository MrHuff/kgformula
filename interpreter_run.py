from kgformula.utils import simulation_object
if __name__ == '__main__':
    h_0_str = 'ground_truth=H_0_y_a=0.0_y_b=0.0_z_a=0.0_z_b=0.5_cor=0.5_n=1000_seeds=1000'
    h_1_str = 'ground_truth=H_1_y_a=0.0_y_b=0.5_z_a=0.0_z_b=0.5_cor=0.5_n=1000_seeds=1000'
    h_0_str_10000 = 'ground_truth=H_0_y_a=0.0_y_b=0.0_z_a=0.0_z_b=0.5_cor=0.5_n=10000_seeds=100'
    h_1_str_10000 = 'ground_truth=H_1_y_a=0.0_y_b=0.5_z_a=0.0_z_b=0.5_cor=0.5_n=10000_seeds=100'
    h_0_str_mult_2 = 'beta_xy=[0, 0]_d_X=3_d_Y=3_d_Z=3_n=1000_yz=0.5_beta_XZ=0.11111'
    h_1_str_mult_2 = 'beta_xy=[0, 0.5]_d_X=3_d_Y=3_d_Z=3_n=1000_yz=0.5_beta_XZ=0.11111'
    h_0_str_mult_2_10000 = 'beta_xy=[0, 0.0]_d_X=3_d_Y=3_d_Z=3_n=10000_yz=0.5_beta_XZ=[0.0, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333]'
    h_1_str_mult_2_10000 = 'beta_xy=[0, 0.5]_d_X=3_d_Y=3_d_Z=3_n=10000_yz=0.5_beta_XZ=[0.0, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333]'
    h_0_str_mult_2_big = 'beta_xy=[0, 0]_d_X=3_d_Y=3_d_Z=50_n=1000_yz=0.5_beta_XZ=0.04'
    h_1_str_mult_2_big = 'beta_xy=[0, 0.5]_d_X=3_d_Y=3_d_Z=50_n=1000_yz=0.5_beta_XZ=0.04'
    h_0_str_mult_2_big_10000 = 'beta_xy=[0, 0.0]_d_X=3_d_Y=3_d_Z=50_n=10000_yz=0.5_beta_XZ=[0.0, 0.02, 0.02, 0.02, 0.02]'
    h_1_str_mult_2_big_10000 = 'beta_xy=[0, 0.5]_d_X=3_d_Y=3_d_Z=50_n=10000_yz=0.5_beta_XZ=[0.0, 0.02, 0.02, 0.02, 0.02]'
    for h,seeds in zip([h_0_str_mult_2_big,h_1_str_mult_2_big],[1000,1000]):
        args={
            'data_dir': h,
            'estimate':False,
            'debug_plot':False,
            'seeds':seeds,
            'bootstrap_runs':250,
            'debug_generative_process':False,
            'debug_d_Z':3,
            'est_params' : {'lr': 1e-4,
                      'max_its': 5000,
                      'width': 128,
                      'layers': 4,
                      'mixed': False,
                      'bs_ratio': 1e-2,
                      'kappa': 10,
                      'val_rate':0.01,
                      'n_sample':250,
                      'criteria_limit':0.10,
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