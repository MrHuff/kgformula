from kgformula.test_statistics import  density_estimator,HSIC_independence_test
from kgformula.utils import simulation_object,get_density_plot
import torch
import GPUtil
import seaborn as sns; sns.set()
from density_estimator_comparison import get_w_estimate_and_plot
import pandas as pd
import gpytorch
if __name__ == '__main__':
    torch.manual_seed(1)
    x = torch.randn(*(10,1))
    # torch.manual_seed(1)
    y = torch.randn(*(10,1))
    print(x)
    print(y)
    # idx = torch.randperm(10)
    # ones = torch.ones(*(10,1))
    # kernel = gpytorch.kernels.RBFKernel()
    # k_1 = kernel(x).evaluate()
    # print(k_1)
    # k_2 = kernel(x[idx]).evaluate()
    # print(k_2)
    # k_2_sum = k_2@ones
    # print(k_2_sum)
    # print(k_2_sum.repeat(1,10))
    # k_3 = k_1[idx,:]
    # k_3 = k_3[:,idx]
    # print(k_3==k_2)



    # mse_loss = torch.nn.MSELoss()
    # estimator = 'classifier'
    # device = GPUtil.getFirstAvailable(order='memory')[0]
    # i = 0
    # h_0_str_mult_2 = 'beta_xy=[0, 0.0]_d_X=3_d_Y=3_d_Z=3_n=1000_yz=0.5_beta_XZ=[0.0, 0.25, 0.25, 0.25]'
    #
    # data_dir = h_0_str_mult_2
    # X, Y, Z, w_true = torch.load(f'./{data_dir}/data_seed={i}.pt',map_location=f'cuda:{device}')
    #     # 'ground_truth=H_0_y_a=0.0_y_b=0.0_z_a=0.0_z_b=0.5_cor=0.5_n=1000_seeds=1000'
    #
    #
    # print(X.shape)
    # print(Z.shape)
    # indep = HSIC_independence_test(X,Z,1000)
    # print(indep.p_val)
    # est_params = {'lr': 1e-3,
    #               'max_its': 5000,
    #               'width': 64,
    #               'layers': 4,
    #               'mixed': False,
    #               'bs_ratio': 0.01,
    #               'kappa': 10,
    #               'val_rate':1e-2,
    #               'n_sample':1000,
    #               'criteria_limit':0.25,
    #               'kill_counter':10}
    # w_classification = get_w_estimate_and_plot(X,Z,est_params,estimator,device)
    # with torch.no_grad():
    #     l = mse_loss(w_true,w_classification)/w_true.var()
    #     print(1-l.item())
    # test = torch.load(h_0_str_mult_2+'/hsic_pval_array_seeds=10_estimate=True_estimator=classifier_lr=0.001_max_its=5000_width=64_layers=4_mixed=False_bs_ratio=0.01_kappa=10_val_rate=0.01_n_sample=250_criteria_limit=0.1_kill_counter=10_reg_lambda=0.01_alpha=0.5.pt')
    # test_2 = torch.load(h_0_str_mult_2+'/p_val_array_seeds=10_estimate=True_estimator=classifier_lr=0.001_max_its=5000_width=64_layers=4_mixed=False_bs_ratio=0.01_kappa=10_val_rate=0.01_n_sample=250_criteria_limit=0.1_kill_counter=10_reg_lambda=0.01_alpha=0.5.pt')

    # print(test)
    # print(test_2)