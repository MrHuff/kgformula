import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.stats import kstest

def load_pval_data(file_path,suffix):
    p_val_test = torch.load(file_path+'/p_val_array_'+suffix+'.pt').numpy()
    try:
        hsic_pval = torch.load(file_path+'/hsic_pval_array_'+suffix+'.pt').numpy()
    except Exception as e:
        hsic_pval = []
    return p_val_test,hsic_pval

if __name__ == '__main__':

    h_0_str_mult_2 = './beta_xy=[0, 0.5]_d_X=3_d_Y=3_d_Z=3_n=1000_yz=0.5_beta_XZ=0.3'
    suffix = 'seeds=1000_estimate=False_estimator=classifier_lr=0.0001_max_its=5000_width=128_layers=4_mixed=False_bs_ratio=0.01_kappa=10_val_rate=0.01_n_sample=250_criteria_limit=0.1_kill_counter=10_reg_lambda=0.01_alpha=0.5'
    p_val_test, hsic_pval = load_pval_data(h_0_str_mult_2,suffix)
    hsic_pval_cutoff = [0,1e-4,5e-4,0.01,0.05,0.1,0.15,0.2,0.25,0.5,0.60,0.7,0.8,0.9]
    plt.hist(p_val_test,bins=25)
    plt.show()
    plt.clf()
    try:
        fig, axs = plt.subplots(len(hsic_pval_cutoff)//3+1,3)
        filtered_n = []
        ks_pval = []
        for i,h in enumerate(hsic_pval_cutoff):
            mask = hsic_pval>=h
            n = mask.sum()
            filtered_n.append(n)
            tmp_test_pvals = p_val_test[mask]
            ks_stat, p_val_ks_test = kstest(tmp_test_pvals, 'uniform')
            ks_pval.append(p_val_ks_test)
            axs[i//3][i%3].hist(tmp_test_pvals, bins=25)
            axs[i//3][i%3].set_title(f'filter>={h}, KS-test p={round(p_val_ks_test,3)}', fontsize=6)
        plt.show()
        fig_2, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(hsic_pval_cutoff, filtered_n, 'g-')
        ax2.plot(hsic_pval_cutoff, ks_pval, 'b-')
        ax1.set_xlabel('HSIC filter p_val>=x')
        ax1.set_ylabel('n remaning', color='g')
        ax2.set_ylabel('KS-test p_val', color='b')
        plt.show()
    except Exception as e:
        print('nvm')