import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.max_rows", 999)
pd.set_option("display.max_columns", 90)
pd.set_option('display.expand_frame_repr', False)
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def make_collage(directory,ld_str,nrow,ncol):
    fig, ax= plt.subplots(nrows=nrow,ncols=ncol,figsize=(20,20))

    for i,n in enumerate([1000, 5000, 10000]):
        for j,d_Z in enumerate([1, 3, 50]):
            try:
                load_str = f"{directory}{ld_str}_n={n}_dz={d_Z}.png"
                img = mpimg.imread(load_str)
                ax[i,j].imshow(img)
                ax[i, j].axis('off')
            except Exception as e:
                print(e)
    plt.tight_layout()
    fig.savefig(f"{directory}collage_{ld_str}.png")

def scatter_plot_KS_null_vs_corr(df_name,directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory)
    df = pd.read_csv(df_name, index_col=0)
    df_0 = df[df['beta_xy'] == 0]
    for n in [1000, 5000, 10000]:
        df_sub = df_0[df_0['n'] == n]
        for d_Z in [1, 3, 50]:
            try:
                df_sub_2 = df_sub[df_sub['d_Z'] == d_Z]
                c = df_sub_2['$/beta_{xz}$'].values
                scatter=plt.scatter(df_sub_2['true_w_q_corr'],df_sub_2['KS stat'],c=c)
                legend1 = plt.legend(*scatter.legend_elements(),
                                     loc="upper left", title="beta_xz")
                plt.suptitle(f"est_weights: dz={d_Z} n={n//2}")
                plt.xlabel('corr')
                plt.ylabel('KS stat')
                plt.savefig(f"{directory}corr_vs_KS_n={n}_dz={d_Z}.png")
                plt.clf()
            except Exception as e:
                print(e)
                plt.clf()

def scatter_plots_EFF_KS_null(df_name,directory,est=False):
    df = pd.read_csv(df_name, index_col=0)
    # post_process_h1_all_true_weights.csv
    df = df.sort_values(['beta_xy', 'n', 'KS pval'])
    df_0 = df[df['beta_xy'] == 0]
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory)

    for n in [1000, 5000, 10000]:
        df_sub = df_0[df_0['n'] == n]
        if est:
            n = n // 2
        for d_Z in [1, 3, 50]:
            try:
                df_sub_2 = df_sub[df_sub['d_Z'] == d_Z]
                y = df_sub_2['KS stat']
                x = df_sub_2['EFF est w']
                c = df_sub_2['$/beta_{xz}$'].values
                scatter = plt.scatter(x, y, c=c)
                plt.hlines(0.05, 0, n)
                plt.suptitle(f"true_weights: dz={d_Z} n={n}")
                legend1 = plt.legend(*scatter.legend_elements(),
                                     loc="upper left", title="beta_xz")
                plt.xlabel('ESS')
                plt.ylabel('KS stat')
                plt.savefig(f"{directory}ESS_vs_KS_n={n}_dz={d_Z}.png")
                plt.clf()
            except Exception as e:
                print(e)
                print("prolly doesn't exist, yet!")
                plt.clf()

def scatter_plots_EFF_KSpval_null(df_name,directory,est=False):
    df = pd.read_csv(df_name, index_col=0)
    # post_process_h1_all_true_weights.csv
    df = df.sort_values(['beta_xy', 'n', 'KS pval'])
    df_0 = df[df['beta_xy'] == 0]

    for n in [1000, 5000, 10000]:
        df_sub = df_0[df_0['n'] == n]
        if est:
            n = n // 2
        for d_Z in [1, 3, 50]:
            try:
                df_sub_2 = df_sub[df_sub['d_Z'] == d_Z]
                y = df_sub_2['KS pval']
                x = df_sub_2['EFF est w']
                c = df_sub_2['$/beta_{xz}$'].values
                scatter = plt.scatter(x, y, c=c)
                plt.hlines(0.05, 0, n)
                plt.suptitle(f"true_weights: dz={d_Z} n={n}")
                legend1 = plt.legend(*scatter.legend_elements(),
                                     loc="upper left", title="beta_xz")
                plt.xlabel('ESS')
                plt.ylabel('KS pval')
                plt.savefig(f"{directory}ESS_vs_KSpval_n={n}_dz={d_Z}.png")
                plt.clf()
            except Exception as e:
                print(e)
                print("prolly doesn't exist, yet!")
                plt.clf()

if __name__ == '__main__':
    scatter_plots_EFF_KS_null('job_dir_real.csv','true_weights/')
    scatter_plots_EFF_KSpval_null('job_dir_real.csv','true_weights/')
    scatter_plots_EFF_KS_null('job_dir.csv','estimated_weights/',True)
    scatter_plot_KS_null_vs_corr('job_dir.csv','corr_vs_KS_null/')
    make_collage('true_weights/','ESS_vs_KS',3,3)
    make_collage('true_weights/','ESS_vs_KSpval',3,3)
    make_collage('corr_vs_KS_null/','corr_vs_KS',3,3)

