from kgformula.utils import simulation_object
if __name__ == '__main__':
    args={
        'data_dir':'ground_truth=H_0_y_a=0.0_y_b=0.0_z_a=0.0_z_b=0.5_cor=0.5_n=1000_seeds=1000',
        'estimate':True,
        'debug_plot':False,
        'seeds':1000,
        'bootstrap_runs':250,
        'alpha':0.5,
        'estimator':'kmm',
        'lamb':0,
        'runs':1,
        'test_stat':3,
        'cuda':True
    }
    j = simulation_object(args)
    j.run()