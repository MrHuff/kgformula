from kgformula.test_statistics import weighted_stat,weighted_statistic_new, density_estimator,get_i_not_j_indices
from kgformula.utils import simulation_object,get_density_plot
import torch
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time
import GPUtil
import seaborn as sns; sns.set()

if __name__ == '__main__':

    estimator = 'classifier'
    lamb = 1e-2
    data_dir = 'ground_truth=H_0_y_a=0.0_y_b=0.0_z_a=0.0_z_b=0.5_cor=0.5_n=1000_seeds=1000'
    i = 0
    device = GPUtil.getFirstAvailable(order='memory')[0]
    # X, Y, Z, w_true = torch.load(f'./{data_dir}/data_seed={i}.pt',map_location=f'cuda:{device}')
    # X = X.cpu().flatten().numpy()
    # Z = Z.cpu().flatten().numpy()
    # w = w_true.cpu().flatten().numpy()
    # triang = mtri.Triangulation(X, Z)
    # plt.tricontourf(triang, w)
    # plt.colorbar()
    # plt.show()
    X, Y, Z, w_true = torch.load(f'./{data_dir}/data_seed={i}.pt',map_location=f'cuda:{device}')
    est_params = {'lr': 1e-4,'max_its':5000,'width':64,'layers':2,'mixed':False,'bs_ratio':0.01,'kappa':10,'kill_counter':10}
    d = density_estimator(x=X, z=Z, cuda=True, est_params=est_params, type=estimator, reg_lambda=lamb,device=device)
    get_density_plot(d,X,Z)

    # s = time.time()

    # e = time.time()
    # print('mixed: ',e-s)
    # estimator = 'HSIC_classifier'
    # est_params = {'lr': 1e-3,
    #               'max_its':5000,
    #               'kill_counter':10,
    #               'mixed':False,
    #               'bs_ratio':0.1,
    #               'negative_samples':1000,
    #               'x_params':{'d': 1, 'f': 32, 'k': 1, 'o': 3},
    #               'y_params' :{'d': 1, 'f': 32, 'k': 1, 'o': 3},
    #               'kappa':10,
    #
    #               }
    # d = density_estimator(x=X, z=Z, cuda=True, est_params=est_params, type=estimator, reg_lambda=lamb,device=device)
    # get_density_plot(d,X,Z)
    # e_2 = time.time()

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