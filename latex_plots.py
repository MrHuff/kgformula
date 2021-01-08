import pandas as pd
from generate_data_multivariate import generate_sensible_variables,calc_snr
pd.set_option("display.max_rows", 999)
pd.set_option("display.max_columns", 90)
pd.set_option('display.expand_frame_repr', False)
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import *
rc('text', usetex=True)


def calibration_and_power_plots(directory,csv_file,beta_XZ,est,beta_xy): #Should loop include EFF, KS-val, SNR in title, loop over d_x,n, fix_beta_XZ
    df = pd.read_csv(csv_file)

    df_sub = df[df['$/beta_{xz}$']==beta_XZ]
    df_sub = df_sub[df_sub['nce_style']==est]
    df_sub = df_sub[df_sub['beta_xy']==beta_xy]

    if est == 'real_TRE_Q':
        estimator = 'TRE-Q'
    elif est == 'NCE_Q':
        estimator = 'NCE-Q'
    else:
        estimator = est.replace('_', ' ')

    for d_Z in [1,3,15,50]:
        df_sub_2 = df_sub[df_sub['d_Z']==d_Z]
        try:
            if beta_xy==0:
                y = df_sub_2['KS pval']
                fig_name = f"{directory}/n_vs_KSpval_dz={d_Z}_{est}.png"
                ylab = 'KS pval'
            else:
                y = df_sub_2['p_a=0.05']
                fig_name = f"{directory}/n_vs_pow_dz={d_Z}_{beta_xy}_{est}.png"
                ylab = r'Power $\alpha=0.05$'

            x = df_sub_2['n']
            c = df_sub_2['$c_q$'].values
            scatter = plt.scatter(x, y, c=c,alpha=0.5,marker='*',cmap='Set1')
            plt.hlines(0.05, 0, 10000)
            plt.suptitle(r"Estimator: {est} $\quad d_Z$={dz} $\quad\beta_{XY}={bxy}$".format(dz=d_Z,est=estimator,XY='{XY}',bxy=beta_xy))
            legend1 = plt.legend(*scatter.legend_elements(),title=r'$c_q$')
            plt.xlabel('n')
            plt.ylabel(ylab)
            plt.savefig(fig_name)
            plt.clf()
            print("OK")
        except Exception as e:
            print(e)
            print("prolly doesn't exist, yet!")

def calibration_and_power_plots_2(directory,csv_file,beta_XZ,est,n): #Should loop include EFF, KS-val, SNR in title, loop over d_x,n, fix_beta_XZ
    df = pd.read_csv(csv_file)

    df_sub = df[df['$/beta_{xz}$']==beta_XZ]
    df_sub = df_sub[df_sub['nce_style']==est]
    df_sub = df_sub[df_sub['n']==n]

    if est == 'real_TRE_Q':
        estimator = 'TRE-Q'
    elif est == 'NCE_Q':
        estimator = 'NCE-Q'
    else:
        estimator = est.replace('_', ' ')

    for d_Z in [1,3,15,50]:
        df_sub_2 = df_sub[df_sub['d_Z']==d_Z]
        try:
            y = df_sub_2['p_a=0.05']
            fig_name = f"{directory}/bxy_vs_pow_dz={d_Z}_{n}_{est}.png"
            ylab = r'Power $\alpha=0.05$'

            x = df_sub_2['beta_xy']
            c = df_sub_2['$c_q$'].values
            scatter = plt.scatter(x, y, c=c,alpha=0.5,marker='*',cmap='Set1')
            plt.hlines(0.05, 0, 0.5)
            plt.suptitle(r"Estimator: {est} $\quad d_Z$={dz} $\quad n={n}$".format(dz=d_Z,est=estimator,n=n))
            legend1 = plt.legend(*scatter.legend_elements(),title=r'$c_q$')
            plt.xlabel(r'$\beta_{XY}$')
            plt.ylabel(ylab)
            plt.savefig(fig_name)
            plt.clf()
            print("OK")
        except Exception as e:
            print(e)
            print("prolly doesn't exist, yet!")



if __name__ == '__main__':
    configs = ['real_weights','NCE_Q','real_TRE_Q','rulsif']
    l_a = ['job_rulsif.csv','job_dir_harder_real.csv','job_dir_real.csv','job_dir.csv','job_dir_harder.csv','job_dir_harder_real_2.csv','job_dir_harder_real_3.csv','job_dir_harder_2.csv','job_dir_harder_3.csv']
    for file in l_a:
        dir = file.split('.')[0]+'_plots'
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            shutil.rmtree(dir)
            os.makedirs(dir)
        for e in configs:
            for bxy in [0.0,0.1,0.25,0.5]:
                calibration_and_power_plots(dir,csv_file=file,beta_XZ=0.5,est=e,beta_xy=bxy)

        for e in configs:
            for n in [1000,5000,10000]:
                calibration_and_power_plots_2(dir,csv_file=file,beta_XZ=0.5,est=e,n=n)