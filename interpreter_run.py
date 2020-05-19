from kgformula.utils import simulation_object
import math
if __name__ == '__main__':
    beta_XZ=0.0 #beta_XZ = 0.5, 0.11111, 0.0004
    for n in [100,1000,10000]:
        h_0_str = f'ground_truth=H_0_y_a=0.0_y_b=0.0_z_a=0.0_z_b={beta_XZ}_cor=0.5_n={n}_seeds=100'
        h_1_str = f'ground_truth=H_1_y_a=0.0_y_b=0.5_z_a=0.0_z_b={beta_XZ}_cor=0.5_n={n}_seeds=100'
        h_0_str_mult_2 = f'beta_xy=[0, 0]_d_X=3_d_Y=3_d_Z=3_n={n}_yz=0.5_beta_XZ={beta_XZ}'
        h_1_str_mult_2 = f'beta_xy=[0, 1.0]_d_X=3_d_Y=3_d_Z=3_n={n}_yz=0.5_beta_XZ={beta_XZ}'
        h_0_str_mult_2_big = f'beta_xy=[0, 0]_d_X=3_d_Y=3_d_Z=50_n={n}_yz=0.5_beta_XZ={beta_XZ}'
        h_1_str_mult_2_big = f'beta_xy=[0, 1.0]_d_X=3_d_Y=3_d_Z=50_n={n}_yz=0.5_beta_XZ={beta_XZ}'
        seed_max = 100
        for h in [h_0_str,h_1_str,h_0_str_mult_2,h_1_str_mult_2,h_0_str_mult_2_big,h_1_str_mult_2_big]:#zip([h_0_str_mult_2_big,h_1_str_mult_2_big],[seed_max,seed_max]):
            args={
                'data_dir': h,
                'estimate':True,
                'debug_plot':False,
                'seeds':seed_max,
                'bootstrap_runs':250,
                'debug_generative_process':False,
                'debug_d_Z':3,
                'est_params' : {'lr': 1e-3,
                          'max_its': 5000,
                          'width': int(math.log10(n))*16,
                          'layers': int(math.log10(n)),
                          'mixed': False,
                          'bs_ratio': 10./n,
                          'kappa': 10,
                          'val_rate':10./n,
                          'n_sample':250,
                          'criteria_limit':0.05,
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