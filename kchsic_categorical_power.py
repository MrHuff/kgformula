import os

import pandas as pd

from post_process import *
from scipy.stats import kstest
from create_plots import *

def plot_power(df,dir):
    for d in [1]:
        for alp in [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]:
            subset_3 = df[df['alp']==alp]
            a,b,e = calc_error_bars(subset_3['alp=0.05'],alpha=0.05,num_samples=100)
            plt.plot('n','alp=0.05',data=subset_3,linestyle='--', marker='o',label=r'$\alpha'+f'={alp}$')
            plt.fill_between(subset_3['n'], a, b, alpha=0.1)
        plt.hlines(0.05, 0.1, 0.5)
        plt.legend(prop={'size': 10})
        plt.xticks([1000,5000,10000])
        plt.xlabel(r'$n$')
        plt.ylabel(r'Power $\alpha=0.05$')
        plt.savefig(f'{dir}/pow_plot_bench_{d}.png',bbox_inches = 'tight',pad_inches = 0.05)
        plt.clf()

def calc_power(vec, level=.05):
    n = vec.shape[0]
    pow = np.sum(vec<=level)/n
    return pow

def extract_properties(job_params):
    data_dir = job_params['job_dir']
    n = job_params['n']
    estimator = job_params['estimator']
    mode= job_params['mode']
    qdist = job_params['qdist']
    suffix = f'_qf=rule_qd={qdist}_m={mode}_s={0}_{100}_e={True}_est={estimator}_sp={True}_br={500}_n={n}'
    load_path = job_params['data_dir']+'/'+data_dir+'/'+f'p_val_array{suffix}.pt'
    string_base = job_params['data_dir'].split('_')
    alp = float(string_base[3].split('=')[-1])
    null = string_base[4].split('=')[-1]
    properties= [alp,null,n,estimator,data_dir]
    return properties,load_path

bench_res_dir = '1d_cat_pow_kchsic'
job_dir = 'do_null_binary_all_1d'
if not os.path.exists(bench_res_dir):
    os.makedirs(bench_res_dir)
# print(benchmark_data)

jobs = os.listdir('do_null_binary_all_1d')
df_dat = []
for j in jobs:
    job_params = load_obj(j, folder=f'{job_dir}/')
    props, load_path = extract_properties(job_params)
    df_dat.append(props)
    p_vals = torch.load(load_path).cpu().numpy()
    if props[1] == 'False':
        for lvl in [0.01, 0.05, 0.1]:
            pow = calc_power(p_vals, lvl)
            props.append(pow)
    else:
        _, p_val_ks_test = kstest(p_vals, 'uniform')
        for i in range(3):
            props.append(p_val_ks_test)
    df_dat.append(props)

df = pd.DataFrame(df_dat,columns= [ 'alp','null','n','estimator','data_dir','alp=0.01','alp=0.05','alp=0.1'])
subset = df[(df['null']=='False')&(df['data_dir']=='do_null_binary_all_1d_layers=1_width=32_True')]
plot_power(subset,bench_res_dir)









