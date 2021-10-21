import os
import pandas as pd
from post_process import *
from scipy.stats import kstest
from create_plots import *

dict_method = {'NCE_Q': 'Classifier', 'real_TRE_Q': 'TRE-Q', 'random_uniform': 'random uniform', 'rulsif': 'RuLSIF','real_weights': 'Real weights'}


def calc_power(vec, level=.05):
    n = vec.shape[0]
    pow = np.sum(vec<=level)/n
    return pow


bench_extract_cols = list(str(el) for el in range(1,101))

if __name__ == '__main__':
    df_list = []
    for n in [1000,5000,10000]:
        for beta_xy in [0.0,0.01,0.02,0.03,0.04,0.05]:

            row = pd.read_csv(f"univar_cont_xy={beta_xy}_n={n}.csv")
            p_vals = row[bench_extract_cols].values.astype(float).squeeze()
            pow = calc_power(p_vals)
            df_list.append(['hdm',1,beta_xy,0.75,n,pow])

    df = pd.DataFrame(df_list,columns=['nce_style','d_Z','beta_xy','$/beta_{xz}$','n','p_a=0.05'])
    df.to_csv("cont_hdm_pow.csv")

    df_list = []
    y_index =4
    for n in [1000,5000,10000]:
        for beta_xy in [0.0,0.01,0.02,0.03,0.04,0.05]:

            row = pd.read_csv(f"hdm_fail_cont_xy={beta_xy}_n={n}_y={y_index}.csv")
            p_vals = row[bench_extract_cols].values.astype(float).squeeze()
            pow = calc_power(p_vals)
            df_list.append(['hdm',1,beta_xy,0.75,n,pow])

    df = pd.DataFrame(df_list,columns=['nce_style','d_Z','beta_xy','$/beta_{xz}$','n','p_a=0.05'])
    df.to_csv(f"break_hdm_pow_{y_index}.csv")