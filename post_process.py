import pandas as pd
import torch
import numpy as np
from scipy.stats import kstest
import matplotlib.pyplot as plt
from post_process_plots import get_size_power_of_test,plot_size_power_deviation
def concat_data(PATH,prefix,suffices):
    collection = []
    for el in suffices:
        collection.append(torch.load(PATH+prefix+el+'.pt'))
    return torch.cat(collection,dim=0).numpy()

def return_file_name(a,b,perm):

    _tmp = f'm=Q_s={a}_{b}_e=False_est=NCE_sp=False_p={perm}'
    #f'm=new_s={a}_{b}_e=True_est=NCE_sp=True_lr=0.0001_ma=5000_wi=32_la=4_mi=False_bs=0.001_va=0.01_n_=250_cr=0.05_ki=10_ka=10_m=10000_d_=3_d_=3'
    # f'm=Q_s={a}_{b}_e=True_est=NCE_sp=True_lr=0.0001_ma=5000_wi=32_la=4_mi=False_bs=0.001_va=0.01_n_=250_cr=0.05_ki=10_ka=10_m=10000_d_=3_d_=3'
    #f'm=regular_s={a}_{b}_e=False_est=NCE_sp=False'
    #f'm=new_s={a}_{b}_e=True_est=NCE_sp=True_lr=0.0001_ma=5000_wi=32_la=4_mi=False_bs=0.001_va=0.01_n_=250_cr=0.05_ki=10_ka=10_m=10000_d_=1_d_=1'
    return _tmp

def calc_eff(w):
    return (w.sum()**2)/(w**2).sum()

prefix_pval = 'p_val_array_'
prefix_ref = 'ref_val_array_'
prefix_hsic_pval = 'hsic_pval_array_'

h_0=True
p_val = True
size = True
histogram_true = True
ref_val = False
hsic_val = False
seed_max=100
if __name__ == '__main__':

    df_data = []
    for perm,nr_of_gpus in zip(['Y','X'],[3,4]):
        for beta_xz in [0.0,1e-3,1e-2,0.05,0.1,0.25,0.5,1]:
            #0,0.01,0.1,0.25,0.5
            #0.0, 0.001, 0.011, 0.111
            #0.0,0.004,0.02
                #[0.0,0.001,0.028,0.056,0.111,0.5]:
            for q in [1e-3,1e-2,0.05,0.1,0.25,0.5,1]:
                row = [perm,beta_xz,q]
                residual = seed_max % nr_of_gpus
                interval_size = (seed_max - residual) / nr_of_gpus
                suffices = []
                data_path = f'q={q}_mv_100/beta_xy=[0, 0]_d_X=3_d_Y=3_d_Z=50_n=10000_yz=0.5_beta_XZ={beta_xz}_theta=4_phi=2.0/'
                #f'univariate_100_seeds/Q={q}_gt=H_0_y_a=0.0_y_b=0.0_z_a=0.0_z_b={beta_xz}_cor=0.5_n=10000_seeds=100_4_2.0/'
                #f'q={q}_mv_100/beta_xy=[0, 0]_d_X=3_d_Y=3_d_Z=50_n=10000_yz=0.5_beta_XZ={beta_xz}_theta=4_phi=2.0/'
                PATH = data_path+'layers=4_width=32/'
                    #
                    #f'univariate_100_seeds/Q={q}_gt=H_0_y_a=0.0_y_b=0.0_z_a=0.0_z_b={beta_xz}_cor=0.5_n=10000_seeds=100_4_2.0/'
                for i in range(nr_of_gpus):
                    if i == nr_of_gpus - 1:
                        a = int(i * interval_size)
                        b = int((i + 1) * interval_size + residual)
                    else:
                        a = int(i * interval_size)
                        b = int((i + 1) * interval_size )
                    suffices.append(return_file_name(a,b,perm))

                if histogram_true:
                    X, Y, Z, X_q, _w, w_q = torch.load(data_path+'data_seed=0.pt')
                    w_eff = round(calc_eff(_w).item(),2)
                    w_q_eff = round(calc_eff(w_q).item(),2)

                    plt.hist(_w,250)
                    plt.suptitle(f'true weights w_i, EFF={w_eff}')
                    plt.savefig(PATH+f'w_i_{q}_{beta_xz}'+return_file_name(0,100,perm)+'.jpg')
                    plt.clf()
                    plt.hist(w_q,250)
                    plt.suptitle(f'true weights w_i^q, EFF={w_q_eff}')
                    plt.savefig(PATH+f'w_i^q_{q}_{beta_xz}'+return_file_name(0,100,perm)+'.jpg')
                    plt.clf()
                    row.append(w_q_eff)

                if p_val:
                    pval_dist = concat_data(PATH,prefix_pval,suffices)
                    stat, pval = kstest(pval_dist, 'uniform')
                    ks_test = pd.DataFrame([[stat,pval]],columns=['ks-stat','ks-pval'])
                    ks_test.to_csv(PATH+'fdf_'+return_file_name(0,100,perm)+'.csv')
                    plt.hist(pval_dist,25)
                    plt.savefig(PATH+f'pval_{q}_{beta_xz}'+return_file_name(0,100,perm)+'.jpg')
                    plt.clf()
                    row.append(pval)

                if ref_val:
                    ref_vals = concat_data(PATH,prefix_pval,suffices)
                    plt.hist(ref_vals,25)
                    plt.savefig(PATH+f'refval_{q}_{beta_xz}'+return_file_name(0,100,perm)+'.jpg')
                    plt.clf()

                if size:
                    pval_dist = concat_data(PATH,prefix_pval,suffices)
                    levels = [1e-3,1e-2,0.05,1e-1]
                    str_mode ='size' if h_0 else 'power'
                    results_size = []
                    results_deviation = []
                    for alpha in levels:
                        size,relative_deviation = get_size_power_of_test(alpha,pval_dist,h_0=h_0)
                        results_size.append(size)
                        results_deviation.append(relative_deviation)
                    plot_size_power_deviation(
                        levels,
                        results_size,
                        results_deviation,
                        str_mode,
                        PATH + str_mode+f'_{q}_{beta_xz}' + return_file_name(0, 100,perm) + '.jpg'
                    )
                df_data.append(row)
    df = pd.DataFrame(df_data,columns=['permutation','beta_xz','c_q','EFF w_q','KS pval'])
    df = df.sort_values('EFF w_q',ascending=False)
    df.to_csv('post_process.csv')
    print(df.to_latex())
