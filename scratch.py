from kgformula.test_statistics import weighted_stat,weighted_statistic_new, density_estimator,get_i_not_j_indices
from kgformula.utils import simulation_object
import torch
import numpy as np
from matplotlib import pyplot as plt
import time
import GPUtil
if __name__ == '__main__':
    estimator = 'HSIC_classifier'
    lamb = 1e-2
    data_dir = 'ground_truth=H_0_y_a=0.0_y_b=0.0_z_a=0.0_z_b=0.5_cor=0.5_n=1000_seeds=1000'
    i = 0
    device = GPUtil.getFirstAvailable(order='memory')[0]
    X, Y, Z, _ = torch.load(f'./{data_dir}/data_seed={i}.pt',map_location=f'cuda:{device}')
    # est_params = {'lr': 1e-3,'max_its':10000,'width':512,'layers':1,'auc':0.95,'mixed':True,'bs_ratio':0.1,'negative_samples':1000*10}
    # s = time.time()
    # d = density_estimator(x=X, z=Z, cuda=True, est_params=est_params, type=estimator, reg_lambda=lamb,device=device)
    # e = time.time()
    # print('mixed: ',e-s)
    est_params = {'lr': 1e-3,
                  'max_its':10000,
                  'auc':0.95,
                  'mixed':False,
                  'bs_ratio':0.1,
                  'negative_samples':1000*10,
                  'x_params':{'d': 1, 'f': 12, 'k': 2, 'o': 3},
                  'y_params' :{'d': 1, 'f': 12, 'k': 2, 'o': 3}
                  }
    d = density_estimator(x=X, z=Z, cuda=True, est_params=est_params, type=estimator, reg_lambda=lamb,device=device)
    e_2 = time.time()

    # args = {
    #     'data_dir': 'ground_truth=H_0_y_a=0.0_y_b=0.0_z_a=0.0_z_b=0.5_cor=0.5_n=1000_seeds=1000',
    #     'estimate': True,
    #     'debug_plot': False,
    #     'seeds': 2,
    #     'bootstrap_runs': 250,
    #     'alpha': 0.5,
    #     'estimator': 'kmm',
    #     'lamb': 1e-1,  # Bro, regularization seems to shift the p-value distribution to the left wtf.
    #     'runs': 1,
    #     'test_stat': 3,
    #     'cuda': True
    # }
    # j = simulation_object(args)
    # j.debug_w(lambdas=[0], expected_shape=1000, estimator='truth')
    # j.debug_w(lambdas=[0,1e-5, 0.01, 0.1, 1.0], expected_shape=1000, estimator='kmm')
    # data_dir = 'ground_truth=H_0_y_a=0.0_y_b=0.0_z_a=0.0_z_b=0.5_cor=0.5_n=1000_seeds=1000'
    # X, Y, Z, w = torch.load(f'./{data_dir}/data_seed={0}.pt')
    # plt.scatter(np.arange(0,1000),w.numpy())
    # plt.show()



    # w = d.return_weights()
    # end = time.time()
    # print(end-start)
    # vec = np.arange(0,1000)
    # vec_2 = np.arange(0,int(1000**2),1001)
    # list_np = np.array(np.meshgrid(vec,vec)).T.reshape(-1, 2)
    # list_np = np.delete(list_np,vec_2,axis=0)
    # end_2 = time.time()
    # print(end_2-end)
    # print(list_np.shape)
    #
    # torch_idx = torch.tensor(list_idx).long()
    # torch_idx_x, torch_idx_z = torch_idx.unbind(dim=1)
    #