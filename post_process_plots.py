import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.stats import kstest

def plot_size_power_deviation(levels_alpha,size_y,deviation_y,power_or_size='size'):
    fig_2, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.bar([str(el)for el in levels_alpha], size_y,color='g')
    ax2.plot([str(el)for el in levels_alpha], deviation_y,color='b')
    ax1.set_xlabel('alpha')
    ax1.set_ylabel(power_or_size, color='g')
    ax2.set_ylabel('Relative deviation(%)', color='b')
    plt.show()
    plt.hist(p_val_test,bins=25)
    plt.show()
    plt.clf()

def load_pval_data(file_path,suffix):
    p_val_test = torch.load(file_path+'/p_val_array_'+suffix+'.pt').numpy()
    try:
        hsic_pval = torch.load(file_path+'/hsic_pval_array_'+suffix+'.pt').numpy()
    except Exception as e:
        hsic_pval = []
    return p_val_test,hsic_pval

def get_size_power_of_test(level,p_values,h_0=True):
    total_pvals = len(p_values)
    size = sum(p_values<=level)/total_pvals
    if h_0:
        relative_deviation = (level-size)/level
    else:
        relative_deviation = (1.-size)/level
    return size,relative_deviation

if __name__ == '__main__':

    h_0_str_mult_2 = './ground_truth=H_1_y_a=0.0_y_b=0.5_z_a=0.0_z_b=0.0_cor=0.5_n=100_seeds=100'
    suffix = 'seeds=100_estimate=True_estimator=classifier_lr=0.001_max_its=5000_width=32_layers=2_mixed=False_bs_ratio=0.1_kappa=10_val_rate=0.1_n_sample=250_criteria_limit=0.05_kill_counter=10_reg_lambda=0.01_alpha=0.5'
    p_val_test, hsic_pval = load_pval_data(h_0_str_mult_2,suffix)
    h_0=False
    print(p_val_test.max())
    print(p_val_test.mean())
    hsic_pval_cutoff = [0,0.05]
    levels_alpha = [0.01,0.025,0.05,0.1]
    size_y = []
    deviation_y = []
    for l in levels_alpha:
        size, deviation=get_size_power_of_test(l,p_val_test,h_0)
        size_y.append(size)
        deviation_y.append(deviation)
    plot_size_power_deviation(levels_alpha,size_y,deviation_y,'size' if h_0 else 'power')
    try:
        fig, axs = plt.subplots(2,1)
        filtered_n = []
        ks_pval = []
        for i,h in enumerate(hsic_pval_cutoff):
            mask = hsic_pval>=h
            n = mask.sum()
            filtered_n.append(n)
            tmp_test_pvals = p_val_test[mask]
            ks_stat, p_val_ks_test = kstest(tmp_test_pvals, 'uniform')
            ks_pval.append(p_val_ks_test)
            axs[i].hist(tmp_test_pvals, bins=25)
            axs[i].set_title(f'filter>={h}, KS-test p={round(p_val_ks_test,3)}', fontsize=6)
        plt.show()
        plt.clf()
        fig_2, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.bar([str(el) for el in hsic_pval_cutoff], filtered_n,color= 'g')
        ax2.plot([str(el) for el in hsic_pval_cutoff], ks_pval, color='b')
        ax1.set_xlabel('HSIC filter p_val>=x')
        ax1.set_ylabel('n remaning', color='g')
        ax2.set_ylabel('KS-test p_val', color='b')
        plt.show()
        plt.clf()

    except Exception as e:
        print(e)
        print('nvm')