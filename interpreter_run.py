from kgformula.utils import simulation_object
if __name__ == '__main__':
    h_0_str = 'ground_truth=H_0_y_a=0.0_y_b=0.0_z_a=0.0_z_b=0.5_cor=0.5_n=1000_seeds=1000'
    h_1_str = 'ground_truth=H_1_y_a=0.0_y_b=0.5_z_a=0.0_z_b=0.5_cor=0.5_n=1000_seeds=1000'
    args={
        'data_dir': h_1_str,
        'estimate':True,
        'debug_plot':False,
        'seeds':100,
        'bootstrap_runs':250,
        'est_params' : {'lr': 1e-2,
                  'max_its': 2000,
                  'width': 32,
                  'layers': 4,
                  'mixed': False,
                  'bs_ratio': 0.01,
                  'kappa': 10,
                  'val_rate':0.1,
                  'n_sample':250,
                  'criteria_limit':0.25},
        'estimator':'classifier',
        'runs':1,
        'cuda':True,
    }
    j = simulation_object(args)
    j.run()