import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.stats import kstest


def plot_size_power_deviation(levels_alpha,size_y,deviation_y,power_or_size='size',path='./'):
    fig_2, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.bar([str(el)for el in levels_alpha], size_y,color='g')
    ax2.plot([str(el)for el in levels_alpha], deviation_y,color='b', linestyle='-', marker='o')
    ax1.set_xlabel('alpha')
    ax1.set_ylabel(power_or_size, color='g')
    ax2.set_ylabel('Relative deviation(%)', color='b')
    plt.savefig(path+'/size_power_plot.png')
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

def big_func(h_0_str_mult_2,suffix,h_0,hsic_pval_cutoff,levels_alpha):
    # h_0_str_mult_2 = './ground_truth=H_1_y_a=0.0_y_b=0.5_z_a=0.0_z_b=0.0_cor=0.5_n=100_seeds=100'
    # suffix = 'seeds=100_estimate=True_estimator=classifier_lr=0.001_max_its=5000_width=32_layers=2_mixed=False_bs_ratio=0.1_kappa=10_val_rate=0.1_n_sample=250_criteria_limit=0.05_kill_counter=10_reg_lambda=0.01_alpha=0.5'
    # h_0 = False
    # hsic_pval_cutoff = [0, 0.05]
    # levels_alpha = [0.01, 0.025, 0.05, 0.1]

    p_val_test, hsic_pval = load_pval_data(h_0_str_mult_2, suffix)
    plt.hist(p_val_test, bins=25)
    plt.savefig(h_0_str_mult_2 + '/p_value_distribution.png')
    plt.clf()
    print(p_val_test.max())
    print(p_val_test.mean())
    size_y = []
    deviation_y = []
    for l in levels_alpha:
        size, deviation = get_size_power_of_test(l, p_val_test, h_0)
        size_y.append(size)
        deviation_y.append(deviation)
    plot_size_power_deviation(levels_alpha, size_y, deviation_y, 'size' if h_0 else 'power', h_0_str_mult_2)
    try:
        fig, axs = plt.subplots(2, 1,figsize=(5,7.5))
        fig.tight_layout()
        filtered_n = []
        ks_pval = []
        for i, h in enumerate(hsic_pval_cutoff):
            mask = hsic_pval >= h
            n = mask.sum()
            filtered_n.append(n)
            tmp_test_pvals = p_val_test[mask]
            ks_stat, p_val_ks_test = kstest(tmp_test_pvals, 'uniform')
            ks_pval.append(p_val_ks_test)
            axs[i].hist(tmp_test_pvals, bins=25)
            axs[i].set_title(f'filter>={h}, KS-test p={round(p_val_ks_test, 3)}', fontsize=6)
            axs[i].set_xlim(0,1)
        plt.savefig(h_0_str_mult_2 + '/hsic_effect.png')
        plt.clf()
        fig_2, ax1 = plt.subplots(figsize=(8,5))
        ax2 = ax1.twinx()
        ax1.bar([str(el) for el in hsic_pval_cutoff], filtered_n, color='g')
        ax2.plot([str(el) for el in hsic_pval_cutoff], ks_pval, color='b', linestyle='-', marker='o')
        ax1.set_xlabel('HSIC filter p_val>=x')
        ax1.set_ylabel('n remaning', color='g')
        ax2.set_ylabel('KS-test p_val', color='b')
        plt.savefig(h_0_str_mult_2 + '/hsic_effect_2.png')
        plt.clf()

    except Exception as e:
        print(e)
        print('nvm')
    return size_y

def plot_size_power_all(folder,size_pow_n,n_s,levels_alpha,size_or_power):
    total = np.stack(size_pow_n,axis=0)
    print(total)
    for i,l in enumerate(levels_alpha):
        if total.ndim==1:
            y = total
        else:
            y = total[:,i]
        plt.bar([str(el) for el in n_s],y)
        plt.axhline(y=l,color='r',linestyle='dashed')
        plt.title(f'alpha={l}')
        plt.xlabel('n')
        plt.ylabel(f'estimated {size_or_power}')
        plt.savefig(folder+f'/alpha={l}_size_power={size_or_power}.png')
        plt.clf()

if __name__ == '__main__':


    suffix = [

        'seeds=100_estimate=True_estimator=classifier_lr=0.001_max_its=5000_width=32_layers=2_mixed=False_bs_ratio=0.1_kappa=10_val_rate=0.1_n_sample=250_criteria_limit=0.05_kill_counter=10_reg_lambda=0.01_alpha=0.5',
        'seeds=100_estimate=True_estimator=classifier_lr=0.001_max_its=5000_width=32_layers=3_mixed=False_bs_ratio=0.01_kappa=10_val_rate=0.01_n_sample=250_criteria_limit=0.05_kill_counter=10_reg_lambda=0.01_alpha=0.5',
        'seeds=100_estimate=True_estimator=classifier_lr=0.001_max_its=5000_width=32_layers=4_mixed=False_bs_ratio=0.001_kappa=10_val_rate=0.01_n_sample=250_criteria_limit=0.05_kill_counter=10_reg_lambda=0.01_alpha=0.5'
        # 'seeds=100_estimate=True_estimator=classifier_lr=0.001_max_its=5000_width=32_layers=2_mixed=False_bs_ratio=0.1_kappa=10_val_rate=0.1_n_sample=250_criteria_limit=0.05_kill_counter=10_reg_lambda=0.01_alpha=0.5',
        # 'seeds=100_estimate=True_estimator=classifier_lr=0.001_max_its=5000_width=48_layers=3_mixed=False_bs_ratio=0.01_kappa=10_val_rate=0.01_n_sample=250_criteria_limit=0.05_kill_counter=10_reg_lambda=0.01_alpha=0.5',
        # 'seeds=100_estimate=True_estimator=classifier_lr=0.001_max_its=5000_width=512_layers=4_mixed=False_bs_ratio=0.001_kappa=10_val_rate=0.01_n_sample=250_criteria_limit=0.05_kill_counter=10_reg_lambda=0.01_alpha=0.5'
        # 'seeds=100_estimate=True_estimator=classifier_lr=0.001_max_its=5000_width=64_layers=4_mixed=False_bs_ratio=0.001_kappa=10_val_rate=0.001_n_sample=250_criteria_limit=0.05_kill_counter=10_reg_lambda=0.01_alpha=0.5'
    ]
    h_0 = False
    hsic_pval_cutoff = [0, 0.05]
    levels_alpha = [0.01, 0.025, 0.05, 0.1] if h_0 else [1.0]
    n_s= [100,1000,10000]
    size_pow_n = []

    for n,suffix in zip(n_s,suffix):
        base = f'./beta_xy=[0, 1.0]_d_X=3_d_Y=3_d_Z=3_n={n}_yz=0.5_beta_XZ=0.03333_theta=8_phi=2.83'
        # base = f'./ground_truth=H_1_y_a=0.0_y_b=0.5_z_a=0.0_z_b=0.5_cor=0.5_n={n}_seeds=100'
        # base = f'./ground_truth=H_1_y_a=0.0_y_b=0.5_z_a=0.0_z_b=0.0_cor=0.5_n={n}_seeds=100'
        size_pow_y = big_func(base, suffix, h_0, hsic_pval_cutoff, levels_alpha)
        size_pow_n.append(size_pow_y)
    plot_size_power_all(base,size_pow_n,n_s,levels_alpha,'size' if h_0 else 'power')

