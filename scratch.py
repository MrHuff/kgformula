from kgformula.test_statistics import  density_estimator,HSIC_independence_test
from kgformula.utils import simulation_object,get_density_plot
import torch
import GPUtil
import seaborn as sns; sns.set()
from density_estimator_comparison import get_w_estimate_and_plot
if __name__ == '__main__':
    mse_loss = torch.nn.MSELoss()
    estimator = 'classifier'
    data_dir = 'ground_truth=H_0_y_a=0.0_y_b=0.0_z_a=0.0_z_b=0.5_cor=0.5_n=1000_seeds=1000'
    i = 0
    device = GPUtil.getFirstAvailable(order='memory')[0]
    X, Y, Z, w_true = torch.load(f'./{data_dir}/data_seed={i}.pt',map_location=f'cuda:{device}')

    # indep = HSIC_independence_test(X,Z,500)
    # print(indep.p_val)
    est_params = {'lr': 1e-3,
                  'max_its': 2000,
                  'width': 32,
                  'layers': 4,
                  'mixed': False,
                  'bs_ratio': 0.01,
                  'kappa': 10,
                  'kill_counter': 10,
                  'val_rate':0.1,
                  'n_sample':250,
                  'criteria_limit':0.25}
    w_classification = get_w_estimate_and_plot(X,Z,est_params,estimator,device)
    with torch.no_grad():
        l = mse_loss(w_true,w_classification)/w_true.var()
        print(1-l.item())

