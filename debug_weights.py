from kgformula.utils import simulation_object
if __name__ == '__main__':
    args={
        'data_dir':'ground_truth=H_1_y_a=0.0_y_b=0.5_z_a=0.0_z_b=0.5_cor=0.5_n=1000_seeds=1000',
        'estimate':True,
        'debug_plot':False,
        'seeds':1000,
        'bootstrap_runs':250,
        'alpha':0.5,
        'estimator':'kmm',
        'lamb':1e-1, #Bro, regularization seems to shift the p-value distribution to the left wtf.
        'runs':1,
        'test_stat':3,
        'cuda':True
    }
    j = simulation_object(args)
    j.debug_w(lambdas=[0],expected_shape=1000,estimator='truth')
    j.debug_w(lambdas=[0,1e-5,0.01,0.1,1.0],expected_shape=1000,estimator='kmm')