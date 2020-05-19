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

def get_size_power_of_test(level,p_values):
    total_pvals = len(p_values)
    size = sum(p_values<=level)/total_pvals
    relative_deviation = (level-size)/level
    return size,relative_deviation

if __name__ == '__main__':

    h_0_str_mult_2 = './ground_truth=H_0_y_a=0.0_y_b=0.0_z_a=0.0_z_b=0.5_cor=0.5_n=1000_seeds=1000'
    suffix = 'seeds=100_estimate=False_estimator=classifier_lr=0.001_max_its=5000_width=32_layers=4_mixed=False_bs_ratio=0.01_kappa=10_val_rate=0.01_n_sample=250_criteria_limit=0.05_kill_counter=10_reg_lambda=0.01_alpha=0.5'
    p_val_test, hsic_pval = load_pval_data(h_0_str_mult_2,suffix)
    print(p_val_test.max())
    print(p_val_test.mean())

    hsic_pval_cutoff = [0,0.05]
    levels_alpha = [0.01,0.025,0.05,0.1]
    plt.hist(p_val_test,bins=25)
    plt.show()
    plt.clf()
    try:
        fig, axs = plt.subplots(1,2)
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
        print(e)
        print('nvm')