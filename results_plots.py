import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.max_rows", 999)
pd.set_option("display.max_columns", 90)
pd.set_option('display.expand_frame_repr', False)
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def make_collage(directory,ld_str,nrow,ncol,est=False):
    fig, ax= plt.subplots(nrows=nrow,ncols=ncol,figsize=(20,20))

    for i,n in enumerate([1000, 5000, 10000]):
        if est:
            n=n//2
        for j,d_Z in enumerate([ 3,15, 50]):
            try:
                load_str = f"{directory}{ld_str}_n={n}_dz={d_Z}.png"
                img = mpimg.imread(load_str)
                ax[i,j].imshow(img)
                ax[i, j].axis('off')
            except Exception as e:
                print(e)
    fig.savefig(f"{directory}collage_{ld_str}.png")
    plt.clf()
def make_collage_power(directory,ld_str,nrow,ncol,p_lims):
    for p_lim in p_lims:
        fig, ax= plt.subplots(nrows=nrow,ncols=ncol,figsize=(20,20))
        for i,beta_xy in enumerate([0.1,0.25,0.5]):
            for j,d_Z in enumerate([3,15, 50]):
                try:
                    load_str = f"{directory}{ld_str}_{d_Z}_{beta_xy}_{p_lim}.png"
                    img = mpimg.imread(load_str)
                    ax[i,j].imshow(img)
                    ax[i, j].axis('off')
                except Exception as e:
                    print(e)
        fig.savefig(f"{directory}collage_{ld_str}_{p_lim}.png")
        plt.clf()
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
        for d_Z in [3,15, 50]:
            try:
                df_sub_2 = df_sub[df_sub['d_Z'] == d_Z]
                c = df_sub_2['$/beta_{xz}$'].values
                scatter=plt.scatter(df_sub_2['true_w_q_corr'],df_sub_2['KS stat'],c=c)
                legend1 = plt.legend(*scatter.legend_elements(),
                                     loc="upper left", title="beta_xz")
                plt.suptitle(f"est_weights: dz={d_Z} n={n//2}")
                plt.xlabel('corr')
                plt.ylabel('KS stat')
                plt.savefig(f"{directory}corr_vs_KS_n={n//2}_dz={d_Z}.png")
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
        for d_Z in [3,15, 50]:
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
        for d_Z in [ 3,15, 50]:
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

def power_plots(df_name,directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory)
    df = pd.read_csv(df_name, index_col=0)
    for d_Z in [3,15, 50]:
        df_sub = df[df['d_Z'] == d_Z]

        for beta_xy in [0.1,0.25,0.5]:
            df_sub_2 = df_sub[df_sub['beta_xy'] == beta_xy]
            for p_lim in ['p_a=0.001', 'p_a=0.01', 'p_a=0.05', 'p_a=0.1']:
                try:
                    alpha = float(p_lim.strip('p_a='))
                    y = df_sub_2[p_lim]
                    x = df_sub_2['n']
                    c = df_sub_2['$/beta_{xz}$'].values
                    scatter = plt.scatter(x, y, c=c)
                    plt.hlines(alpha, 0, 10000)
                    plt.suptitle(f"power: dz={d_Z} beta_xy={beta_xy} alpha={p_lim}")
                    legend1 = plt.legend(*scatter.legend_elements(),
                                         loc="lower right", title="beta_xz")
                    plt.xlabel('n')
                    plt.ylabel('power')
                    plt.savefig(f"{directory}n_vs_power_{d_Z}_{beta_xy}_{p_lim}.png")
                    plt.clf()
                except Exception as e:
                    print(e)
                    print("prolly doesn't exist, yet!")
                    plt.clf()

def generate_plots(csv,fold_name,nrow,ncol):
    power_plots(csv,f'{fold_name}_power/')
    make_collage_power(f'{fold_name}_power/','n_vs_power',nrow,ncol,['p_a=0.001', 'p_a=0.01', 'p_a=0.05', 'p_a=0.1'])
    scatter_plots_EFF_KS_null(csv,f'{fold_name}/')
    scatter_plots_EFF_KSpval_null(csv,f'{fold_name}/')
    make_collage(f'{fold_name}/','ESS_vs_KS',nrow,ncol)


if __name__ == '__main__':
    generate_plots('job_dir_harder_real_3.csv',"harder_true_weights_3",3,3)
    generate_plots('job_dir_harder_3.csv',"harder_est_weights_3",3,3)
    # power_plots('job_dir_real.csv','true_weights_power/')
    # power_plots('job_dir.csv','est_weights_power/')
    # make_collage_power('true_weights_power/','n_vs_power',3,3,['p_a=0.001', 'p_a=0.01', 'p_a=0.05', 'p_a=0.1'])
    # make_collage_power('est_weights_power/','n_vs_power',3,3,['p_a=0.001', 'p_a=0.01', 'p_a=0.05', 'p_a=0.1'])
    # scatter_plots_EFF_KS_null('job_dir_real.csv','true_weights/')
    # scatter_plots_EFF_KSpval_null('job_dir_real.csv','true_weights/')
    # scatter_plots_EFF_KS_null('job_dir.csv','estimated_weights/',True)
    # scatter_plots_EFF_KSpval_null('job_dir.csv','estimated_weights/',True)
    # scatter_plot_KS_null_vs_corr('job_dir.csv','corr_vs_KS_null/')
    # make_collage('true_weights/','ESS_vs_KS',3,3)
    # make_collage('true_weights/','ESS_vs_KSpval',3,3)
    # make_collage('estimated_weights/','ESS_vs_KS',3,3,True)
    # make_collage('estimated_weights/','ESS_vs_KSpval',3,3,True)
    # make_collage('corr_vs_KS_null/','corr_vs_KS',3,3,True)

