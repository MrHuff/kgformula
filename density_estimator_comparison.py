from kgformula.test_statistics import weighted_stat,weighted_statistic_new, density_estimator,get_i_not_j_indices
from kgformula.utils import simulation_object,get_density_plot
import torch
import numpy as np
import GPUtil

def get_w_estimate_and_plot(X,Z,est_params,estimator,device):
    d = density_estimator(x=X, z=Z, cuda=True, est_params=est_params, type=estimator,  device=device)
    get_density_plot(d, X, Z)
    return d.return_weights()

if __name__ == '__main__':

    mse_loss = torch.nn.MSELoss()
    data_dir = 'ground_truth=H_0_y_a=0.0_y_b=0.0_z_a=0.0_z_b=0.5_cor=0.5_n=1000_seeds=1000'
    i = 0
    device = GPUtil.getFirstAvailable(order='memory')[0]
    X, Y, Z, w_true = torch.load(f'./{data_dir}/data_seed={i}.pt', map_location=f'cuda:{device}')

    estimator = 'classifier'
    est_params = {'lr': 1e-3, 'max_its': 2000, 'width': 32, 'layers': 4, 'mixed': False, 'bs_ratio': 0.01, 'kappa': 10,
                  'kill_counter': 5}
    w_classification = get_w_estimate_and_plot(X,Z,est_params,estimator,device)
    with torch.no_grad():
        l = mse_loss(w_true,w_classification)/w_true.var()
        print(1-l.item())


    estimator = 'semi'
    est_params = {}
    w_semi = get_w_estimate_and_plot(X, Z, est_params, estimator, device)
    with torch.no_grad():
        l = mse_loss(w_true,w_semi)/w_true.var()
        print(1-l.item())

    estimator = 'linear'
    est_params = {'reg_lambda':1e-2,'alpha':0.5}
    w_linear = get_w_estimate_and_plot(X, Z, est_params, estimator, device)
    with torch.no_grad():
        l = mse_loss(w_true,w_linear)/w_true.var()
        print(1-l.item())

    estimator = 'gp'
    est_params = {'reg_lambda':1e-2,'alpha':0.5}
    w_gp = get_w_estimate_and_plot(X, Z, est_params, estimator, device)
    with torch.no_grad():
        l = mse_loss(w_true,w_linear)/w_true.var()
        print(1-l.item())

    estimator = 'kmm'
    est_params = {'reg_lambda':1e-2}
    w_kmm = get_w_estimate_and_plot(X, Z, est_params, estimator, device)
    with torch.no_grad():
        l = mse_loss(w_true,w_kmm)/w_true.var()
        print(1-l.item())



