from kgformula.utils import simulation_object
if __name__ == '__main__':
    args={
        'data_dir':'ground_truth=H_0_y_a=0.0_y_b=0.0_z_a=0.0_z_b=0.5_cor=0.5_n=1000_seeds=1000',
        'estimate':True,
        'debug_plot':False,
        'seeds':1000,
        'bootstrap_runs':250,
        'est_params':{'lr': 1e-3,'max_its':2000,'width':32,'layers':4,'mixed':False,'bs_ratio':0.01,'kappa':10,'kill_counter':5},
        'estimator':'classifier',
        'runs':1,
        'cuda':True,
    }
    j = simulation_object(args)
    j.run()