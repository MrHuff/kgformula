from kgformula.test_statistics import  density_estimator,HSIC_independence_test,hsic_sanity_check_w
from kgformula.utils import simulation_object,get_density_plot
import torch
import GPUtil
import seaborn as sns; sns.set()
from kgformula.utils import get_w_estimate_and_plot,debug_W
import pandas as pd
import gpytorch
from matplotlib import pyplot as plt
if __name__ == '__main__':
    # path ='beta_xy=[0, 0]_d_X=3_d_Y=3_d_Z=3_n=10000_yz=0.5_beta_XZ=0.03333_theta=8_phi=2.83'
    # file = '/p_val_array_seeds=100_estimate=True_estimator=classifier_lr=0.001_max_its=5000_width=32_layers=4_mixed=False_bs_ratio=0.001_kappa=10_val_rate=0.01_n_sample=250_criteria_limit=0.05_kill_counter=10_reg_lambda=0.01_alpha=0.5.pt'
    # p = torch.load(path+file)
    # plt.hist(p.numpy(),bins=25)
    # plt.show()
    # plt.clf()

    mse_loss = torch.nn.MSELoss()
    estimator = 'TRE'
    device = GPUtil.getFirstAvailable(order='memory')[0]
    i = 0
    n = 10000
    data_dir ='multivariate_1000' #'multivariate_1000' #univariate_1000_seeds
    X, Y, Z, w_true = torch.load(f'./{data_dir}/beta_xy=[0, 0]_d_X=3_d_Y=3_d_Z=3_n={n}_yz=0.5_beta_XZ=0.5_theta=12_phi=6.86/data_seed={i}.pt',map_location=f'cuda:{device}')
    # get_density_plot(w_true, X, Z)
    print(w_true.shape)
    #'ground_truth=H_0_y_a=0.0_y_b=0.0_z_a=0.0_z_b=0.5_cor=0.5_n={n}_seeds=1000_0.9_0.81'
    #beta_xy=[0, 0]_d_X=3_d_Y=3_d_Z=3_n={n}_yz=0.5_beta_XZ=0.5_theta=12_phi=6.86
    W_true = w_true.unsqueeze(-1)@w_true.unsqueeze(-1).t()
    # print(W_true.shape)
    debug_W(w_true,'w_true')
    # debug_W(W_true,'W_true')

    indep = HSIC_independence_test(X,Z,1000)
    sanity_pval= hsic_sanity_check_w(w_true,X,Z,1000)
    print(indep.p_val)
    print(sanity_pval)
    #
    est_params = {'lr': 1e-4,
                  'max_its': 5000,
                  'width': 32,
                  'layers': 2,
                  'mixed': False,
                  'bs_ratio': 1e-2,
                  'kappa': 10,
                  'val_rate':1e-2,
                  'n_sample':1000,
                  'criteria_limit':0.25,
                  'kill_counter':10,
                  'depth_main':2,
                  'depth_task':2,
                  'outputs': [1,1,1,1,1]}
    w_classification = get_w_estimate_and_plot(X,Z,est_params,estimator,device)
    W_classification = w_classification.unsqueeze(-1)@w_classification.unsqueeze(-1).t()
    debug_W(w_classification,'w_classification')
    # debug_W(W_classification,'W_classification')
    print(w_true)
    print(w_classification)

    with torch.no_grad():
        l = mse_loss(w_true,w_classification)/w_true.var()
        print(1-l.item())
    # test = torch.load(h_0_str_mult_2+'/hsic_pval_array_seeds=10_estimate=True_estimator=classifier_lr=0.001_max_its=5000_width=64_layers=4_mixed=False_bs_ratio=0.01_kappa=10_val_rate=0.01_n_sample=250_criteria_limit=0.1_kill_counter=10_reg_lambda=0.01_alpha=0.5.pt')
    # test_2 = torch.load(h_0_str_mult_2+'/p_val_array_seeds=10_estimate=True_estimator=classifier_lr=0.001_max_its=5000_width=64_layers=4_mixed=False_bs_ratio=0.01_kappa=10_val_rate=0.01_n_sample=250_criteria_limit=0.1_kill_counter=10_reg_lambda=0.01_alpha=0.5.pt')
    #
    # # print(test)
    # # print(test_2)