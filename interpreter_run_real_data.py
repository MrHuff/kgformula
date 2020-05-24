from kgformula.utils import simulation_object
import GPUtil
import math
if __name__ == '__main__':
    cuda = True
    if cuda:
        device = GPUtil.getFirstAvailable(order='memory')[0]
    else:
        device = 'cpu'
    n = 10000
    val_rate = max(1e-2, 10. / n)
    print(val_rate)
    width = [32,128]
    for i,h in enumerate(['infant_mortality_1','covid_19_1']):#zip([h_0_str_mult_2_big,h_1_str_mult_2_big],[seed_max,seed_max]):
        args={
            'device':device,
            'data_dir': h,
            'estimate':True,
            'debug_plot':False,
            'seeds':1,
            'bootstrap_runs':250,
            'debug_generative_process':False,
            'debug_d_Z':3,
            'est_params' : {'lr': 1e-3,
                            'max_its': 10000,
                            'width': width[i],
                            'layers': 4,
                            'mixed': False,
                            'bs_ratio': 10./n,
                            'kappa': 10,
                            'val_rate':val_rate,
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
        del j
            #Figure out what the hell is going on?! why would it work with "X"?!