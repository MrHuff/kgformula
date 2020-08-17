import pandas as pd
import torch
import numpy as np
from scipy.stats import kstest
import matplotlib.pyplot as plt
def concat_data(PATH,prefix,suffices):
    collection = []
    for el in suffices:
        collection.append(torch.load(PATH+prefix+el+'.pt'))
    return torch.cat(collection,dim=0).numpy()

def return_file_name(a,b):
    _tmp = f'm=Q_s={a}_{b}_e=True_est=NCE_sp=True_lr=0.0001_ma=5000_wi=32_la=4_mi=False_bs=0.001_va=0.01_n_=250_cr=0.05_ki=10_ka=10_m=10000_d_=1_d_=1'
    #f'm=regular_s={a}_{b}_e=False_est=NCE_sp=False'
        # f'm=regular_s={a}_{b}_e=True_est=NCE_sp=True_lr=0.0001_ma=5000_wi=32_la=4_mi=False_bs=0.001_va=0.01_n_=250_cr=0.05_ki=10_ka=10_m=10000_d_=1_d_=1'
    return _tmp


prefix_pval = 'p_val_array_'
prefix_ref = 'ref_val_array_'
prefix_hsic_pval = 'hsic_pval_array_'

p_val = True
ref_val = False
hsic_val = False

seed_max = 100
nr_of_gpus = 4
residual = seed_max % nr_of_gpus
interval_size = (seed_max - residual) / nr_of_gpus
suffices = []
PATH ='univariate_100_seeds/Q=0.01_gt=H_0_y_a=0.0_y_b=0.0_z_a=0.0_z_b=0.5_cor=0.5_n=10000_seeds=100_4_2.0/layers=4_width=32/'
for i in range(nr_of_gpus):
    if i == nr_of_gpus - 1:
        a = int(i * interval_size)
        b = int((i + 1) * interval_size + residual)
    else:
        a = int(i * interval_size)
        b = int((i + 1) * interval_size )
    suffices.append(return_file_name(a,b))

if __name__ == '__main__':
    if p_val:
        pval_dist = concat_data(PATH,prefix_pval,suffices)
        stat, pval = kstest(pval_dist, 'uniform')
        ks_test = pd.DataFrame([[stat,pval]],columns=['ks-stat','ks-pval'])
        ks_test.to_csv(PATH+'fdf_'+return_file_name(0,100)+'.csv')
        plt.hist(pval_dist,25)
        plt.savefig(PATH+'pval_'+return_file_name(0,100)+'.jpg')
    if ref_val:
        ref_vals = concat_data(PATH,prefix_pval,suffices)
        plt.hist(ref_vals,25)
        plt.savefig(PATH+'refval_'+return_file_name(0,100)+'.jpg')






