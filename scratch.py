from kgformula.test_statistics import weighted_stat,weighted_statistic_new, density_estimator,get_i_not_j_indices
import torch
import numpy as np

import time
if __name__ == '__main__':
    alpha=0
    estimator = 'kmm'
    lamb = 1e-2
    data_dir = 'simulated_do_null_fixed'
    i = 0
    device = 0
    X, Y, Z, _ = torch.load(f'./{data_dir}/data_seed={i}.pt',map_location=f'cuda:{0}')
    start = time.time()
    d = density_estimator(x=X, z=Z, cuda=True, alpha=alpha, type=estimator, reg_lambda=lamb)
    w = d.return_weights()
    end = time.time()
    print(end-start)


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