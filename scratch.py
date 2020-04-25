from kgformula.test_statistics import  density_estimator
from kgformula.utils import simulation_object,get_density_plot
import torch
import GPUtil
import seaborn as sns; sns.set()

if __name__ == '__main__':

    estimator = 'classifier'
    data_dir = 'ground_truth=H_0_y_a=0.0_y_b=0.0_z_a=0.0_z_b=0.5_cor=0.5_n=1000_seeds=1000'
    i = 0
    device = GPUtil.getFirstAvailable(order='memory')[0]
    X, Y, Z, w_true = torch.load(f'./{data_dir}/data_seed={i}.pt',map_location=f'cuda:{device}')
    est_params = {'lr': 1e-3,
                  'max_its': 2000,
                  'width': 32,
                  'layers': 4,
                  'mixed': False,
                  'bs_ratio': 0.05,
                  'kappa': 10,
                  'kill_counter': 10,
                  'val_rate':0.01}
    d = density_estimator(x=X, z=Z, cuda=True, est_params=est_params, type=estimator,device=device)
    get_density_plot(d,X,Z)

