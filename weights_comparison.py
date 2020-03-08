from kgformula.test_statistics import weighted_stat,weighted_statistic_new,wild_bootstrap_deviance, density_estimator
from kgformula.fixed_do_samplers import simulate_xyz
import os
import torch
import tqdm
from kgformula.utils import hypothesis_acceptance,calculate_pval
import GPUtil
from matplotlib import pyplot as plt
from scipy.stats import kstest
import pandas as pd

def get_mean_and_std(data):
    with torch.no_grad():
        mean = data.mean(dim=0)
        median,_ = data.median(dim=0)
        std = data.var(dim=0)**0.5
        x = [i for i in range(1,data.shape[1]+1)]
        return x, mean.cpu().numpy(),std.cpu().numpy(),median.cpu().numpy()

def plot_and_save(x,mean,std,median,data_dir,name='xyz',alpha=0,reg=0):
    plt.errorbar(x,mean,yerr=std)
    plt.savefig(f'./{data_dir}/{name}_{alpha}_{reg}.png')
    plt.clf()
    plt.scatter(x,median)
    plt.savefig(f'./{data_dir}/{name}_{alpha}_{reg}_median_scatter.png')
    plt.clf()
def do_error_plot(data,data_dir,name,alpha,reg):
    x, mean,std,median = get_mean_and_std(data)
    plot_and_save(x,mean,std,median,data_dir,name,alpha,reg)

if __name__ == '__main__':
    #Log p-values, add vector of p-values
    beta = {'y':[0,0],'z':[0,0.5]}
    cor = 0.5
    data_dir = 'simulated_do_null_fixed_estimates'
    if not os.path.exists(f'./{data_dir}/'):
        os.mkdir(f'./{data_dir}/')
        seeds = 1000
        for i in range(seeds):
            X, Y, Z, w = simulate_xyz(n=1000, beta=beta, cor=cor, fam=1, oversamp=10,seed=i)
            torch.save((X,Y,Z,w),f'./{data_dir}/data_seed={i}.pt')
    plot = False
    test_stat = 2
    n=1000
    seeds = 1000
    bins = 25
    device = GPUtil.getFirstAvailable()[0]
    runs = 1
    ks_data = []
    for alpha in [0,0.5,1]:
        for lamb in [1e-3,1e-2,1e-1]:
            dmw_true = []
            dmw_semi = []
            dmw_gp = []
            dmw_linear = []
            for i in tqdm.trange(seeds):
                X,Y,Z,w_true = torch.load(f'./{data_dir}/data_seed={i}.pt')
                if X.shape[0]!=n:
                    print(X.shape[0])
                    continue
                else:
                    if plot:
                        plt.scatter(Z.numpy(),X.numpy())
                        plt.show()
                    #Cheating case
                    # d = density_estimator(numerator_sample=X,denominator_sample=Z,cuda=True,alpha=alpha,type='semi',reg_lambda=lamb)
                    # w_semi = d.return_weights()
                    d = density_estimator(numerator_sample=X,denominator_sample=Z,cuda=True,alpha=alpha,type='gp',reg_lambda=lamb)
                    w_gp = d.return_weights()
                    d = density_estimator(numerator_sample=X,denominator_sample=Z,cuda=True,alpha=alpha,type='linear',reg_lambda=lamb)
                    w_linear = d.return_weights()
                    # dmw_true.append(w_true)
                    # dmw_semi.append(w_semi)
                    dmw_gp.append(w_gp)
                    dmw_linear.append(w_linear)
            # plot_true = torch.stack(dmw_true,dim=0)
            # plot_semi = torch.stack(dmw_semi,dim=0)
            plot_gp = torch.stack(dmw_gp,dim=0)
            plot_linear = torch.stack(dmw_linear,dim=0)
            # do_error_plot(plot_true,data_dir,'w_true',alpha=alpha,reg=lamb)
            # do_error_plot(plot_semi,data_dir,'w_semi',alpha=alpha,reg=lamb)
            do_error_plot(plot_gp,data_dir,'w_gp',alpha=alpha,reg=lamb)
            do_error_plot(plot_linear,data_dir,'w_linear',alpha=alpha,reg=lamb)







