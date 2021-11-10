import pandas as pd

from post_processing_utils.latex_plots import *
from pylatex import Document, Section, Figure, SubFigure, NoEscape,Command
from pylatex.base_classes import Environment
from pylatex.package import Package
import itertools
import seaborn as sns


dict_method = {'NCE_Q': 'NCE-Q', 'real_TRE_Q': 'TRE-Q', 'random_uniform': 'random uniform', 'rulsif': 'RuLSIF','real_weights':'True weights','hdm':'HDM',
               'NCE_Q_linear':'NCE-Q, linear','random_uniform_linear': 'random uniform linear','rulsif_linear':'RuLSIF linear',
               'real_TRE_Q_linear': 'TRE-Q linear','real_weights_linear': 'True weights linear'}

font_size = 24
plt.rcParams['font.size'] = font_size
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.labelsize'] = font_size
class subfigure(Environment):
    """A class to wrap LaTeX's alltt environment."""
    packages = [Package('subcaption')]
    escape = False
    content_separator = "\n"
    _repr_attributes_mapping = {
        'position': 'options',
        'width': 'arguments',
    }

    def __init__(self, position=NoEscape(r'H'),width=NoEscape(r'0.45\linewidth'), **kwargs):
        """
        Args
        ----
        width: str
            Width of the subfigure itself. It needs a width because it is
            inside another figure.
        """

        super().__init__(options=position,arguments=width, **kwargs)

pd.set_option("display.max_rows", 999)
pd.set_option("display.max_columns", 90)
pd.set_option('display.expand_frame_repr', False)

dim_theta_phi = {
    1:{'d_x':1,'d_y':1,'d_z':1,'theta':2.0,'phi':2.0},
    3:{'d_x':3,'d_y':3,'d_z':3,'theta':4.0,'phi':2.0},
    15:{'d_x':3,'d_y':3,'d_z':15,'theta':8.0,'phi':2.0},
    50:{'d_x':3,'d_y':3,'d_z':15,'theta':16.0,'phi':2.0},
}
dim_theta_phi_hsic = {
    1:{'d_x':1,'d_y':1,'d_z':1,'theta':0.1,'phi':0.9},
}
dim_theta_phi_gcm = {
    1:{'d_x':1,'d_y':1,'d_z':1,'theta':1.0,'phi':2.0},
}
def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    tup = [int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)]
    return [t/255. for t in tup]
color_palette = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
rgb_vals =  [hex_to_rgb(el) for el in color_palette]
print(rgb_vals)
def plot_2_true_weights(csv,dir,mixed=False):
    if isinstance(csv, str):
        df = pd.read_csv(csv, index_col=0)
    else:
        df = csv
    if not os.path.exists(dir):
        os.makedirs(dir)

    d_list = [1,3,15,50] if not mixed else [2,3,15,50]
    for d in d_list:
        subset = df[df['d_Z']==d].sort_values(['beta_xy'])
        for n in [1000, 5000, 10000]:
            subset_2 = subset[subset['n'] == n]
            a,b,e = calc_error_bars(subset_2['p_a=0.05'],alpha=0.05,num_samples=100)
            plt.plot('beta_xy','p_a=0.05',data=subset_2,linestyle='--', marker='o',label=f'$n={n}$')
            plt.fill_between(subset_2['beta_xy'], a, b, alpha=0.1)
            plt.hlines(0.05, 0, subset_2['beta_xy'].max())
        plt.legend(prop={'size': 10})
        plt.xlabel(r'$\beta_{XY}$')
        plt.ylabel('Power')
        plt.savefig(f'{dir}/figure_{d}.png',bbox_inches = 'tight',
    pad_inches = 0.05)
        plt.clf()
    doc = Document(default_filepath=dir)
    with doc.create(Figure(position='H')) as plot:
        with doc.create(subfigure(position='t', width=NoEscape(r'\linewidth'))):
            for i, n in enumerate([1, 3, 15, 50]):
                with doc.create(subfigure(position='H', width=NoEscape(r'0.25\linewidth'))):
                    name = f'$d_Z={n}$'
                    p = f'{dir}/figure_{n}.png'
                    doc.append(Command('centering'))
                    doc.append(r'\rotatebox{0}{\scalebox{0.75}{%s}}' % name)
                    doc.append(r'\includegraphics[width=\linewidth]{%s}'%p)

    doc.generate_tex()

def plot_2_est_weights_test(csv,dir,mixed=False):

    if isinstance(csv,str):
        df = pd.read_csv(csv, index_col=0)
    else:
        df = csv
    if not os.path.exists(dir):
        os.makedirs(dir)

    d_list = [1,3,15,50] if not mixed else [2,3,15,50]
    methods = df['nce_style'].unique().tolist()
    b_Z_list = df['$/beta_{xz}$'].unique().tolist()
    for b_Z in b_Z_list:
        for d in d_list:
            subset = df[(df['d_Z']==d)&(df['$/beta_{xz}$']==b_Z)].sort_values(['beta_xy'])
            for n in [1000, 5000, 10000]:
                subset_2 = subset[subset['n'] == n]
                for method in methods:
                    subset_3 = subset_2[subset_2['nce_style']==method]
                    a,b,e = calc_error_bars(subset_3['p_a=0.05'],alpha=0.05,num_samples=100)
                    format_string = dict_method[method]
                    plt.plot('beta_xy','p_a=0.05',data=subset_3,linestyle='--', marker='o',label=rf'{format_string}')
                    plt.fill_between(subset_3['beta_xy'], a, b, alpha=0.1)
                plt.hlines(0.05, 0, subset_2['beta_xy'].max())
                plt.legend(prop={'size': 10})
                plt.xlabel(r'$\beta_{XY}$')
                plt.ylabel('Power')
                plt.savefig(f'{dir}/figure_{d}_{n}_{b_Z}.png',bbox_inches = 'tight',
            pad_inches = 0.05)
                plt.clf()

def plot_2_est_weights(csv,dir,mixed=False):
    if isinstance(csv,str):
        df = pd.read_csv(csv, index_col=0)
    else:
        df = csv

    if not os.path.exists(dir):
        os.makedirs(dir)

    d_list = [1,3,15,50] if not mixed else [2,3,15,50]
    methods = df['nce_style'].unique().tolist()
    for d in d_list:
        subset = df[df['d_Z']==d].sort_values(['beta_xy'])
        for n in [1000, 5000, 10000]:
            subset_2 = subset[subset['n'] == n]
            for col_index,method in enumerate(methods):
                subset_3 = subset_2[subset_2['nce_style']==method]
                a,b,e = calc_error_bars(subset_3['p_a=0.05'],alpha=0.05,num_samples=100)
                format_string = dict_method[method]
                # plt.plot('beta_xy','p_a=0.05',data=subset_3,linestyle='--', marker='o',label=rf'{format_string}')
                print(format_string)
                plt.plot('beta_xy','p_a=0.05',data=subset_3,linestyle='-',label=rf'{format_string}',c = color_palette[col_index])

                # plt.fill_between(subset_3['beta_xy'], a, b, alpha=0.1)
                plt.fill_between(subset_3['beta_xy'], a, b, alpha=0.1,color=color_palette[col_index])
            plt.hlines(0.05, 0, subset_2['beta_xy'].max())
            # plt.legend(prop={'size': 10})
            plt.xlabel(r'$\beta_{XY}$')
            plt.ylabel('Power')
            plt.savefig(f'{dir}/figure_{d}_{n}.png',bbox_inches = 'tight',
        pad_inches = 0.05)
            plt.clf()
    doc = Document(default_filepath=dir)
    with doc.create(Figure(position='H')) as plot:
        with doc.create(subfigure(position='t', width=NoEscape(r'\linewidth'))):
            for i, n in enumerate([1, 3, 15, 50]):
                if i == 0:
                    with doc.create(subfigure(position='H', width=NoEscape(r'0.04\linewidth'))):
                        # string_append = r'\raisebox{0cm}{\rotatebox[origin=c]{90}{\scalebox{0.75}{}}}' + '%'
                        string_append = '\hfill'
                        doc.append(string_append)
                with doc.create(subfigure(position='H', width=NoEscape(r'0.24\linewidth'))):
                    name = f'$d_Z={n}$'
                    doc.append(Command('centering'))
                    doc.append(r'\rotatebox{0}{\scalebox{0.75}{%s}}' % name)
        counter = 0
        for idx, (i, j) in enumerate(itertools.product([1000, 5000, 10000], [1, 3, 15, 50])):
            if idx % 4 == 0:
                name = f'$n={i}$'
                string_append = r'\raisebox{1.5cm}{\rotatebox[origin=c]{90}{\scalebox{0.75}{%s}}}' % name + '%\n'
            p = f'{dir}/figure_{j}_{i}.png'
            string_append += r'\includegraphics[width=0.24\linewidth]{%s}' % p + '%\n'
            counter += 1
            if counter == 4:
                with doc.create(subfigure(position='H', width=NoEscape(r'\linewidth'))):
                    doc.append(string_append)
                counter = 0
    doc.generate_tex()

def plot_3_est_weights(csv,dir,mixed=True):
    if isinstance(csv, str):
        df = pd.read_csv(csv, index_col=0)
    else:
        df = csv
    if not os.path.exists(dir):
        os.makedirs(dir)

    d_list = [1,3,15,50] if not mixed else [2,15,50]
    methods = df['nce_style'].unique().tolist()
    for d in d_list:
        subset = df[df['d_Z']==d].sort_values(['beta_xy'])
        for n in [1000, 5000, 10000]:
            subset_2 = subset[subset['n'] == n]
            col_index=0
            for _,method in enumerate(methods):
                if method in ['real_weights','random_uniform']:
                    sep_list=[False]
                else:
                    sep_list=[True,False]
                for sep in sep_list:
                    subset_3 = subset_2[ (subset_2['nce_style']==method) & (subset_2['sep']==sep) ]
                    a,b,e = calc_error_bars(subset_3['p_a=0.05'],alpha=0.05,num_samples=100)
                    appendix_string  = ' prod' if sep else ''
                    if method not in ['real_weights','random_uniform']:
                        format_string = dict_method[method] + appendix_string
                    else:
                        format_string = dict_method[method]
                    plt.plot('beta_xy', 'p_a=0.05', data=subset_3, linestyle='-', label=rf'{format_string}',
                             c=color_palette[col_index])

                    # plt.fill_between(subset_3['beta_xy'], a, b, alpha=0.1)
                    plt.fill_between(subset_3['beta_xy'], a, b, alpha=0.1, color=color_palette[col_index])
                    # plt.plot('beta_xy','p_a=0.05',data=subset_3,linestyle='--',label=rf'{format_string}')
                    # plt.fill_between(subset_3['beta_xy'], a, b, alpha=0.1)
                    col_index+=1
            plt.hlines(0.05, 0, subset_2['beta_xy'].max())
            # plt.legend(prop={'size': 10})
            plt.xlabel(r'$\beta_{XY}$')
            plt.ylabel('Power')
            plt.savefig(f'{dir}/figure_{d}_{n}.png',bbox_inches = 'tight',
        pad_inches = 0.05)
            plt.clf()
    doc = Document(default_filepath=dir)
    with doc.create(Figure(position='H')) as plot:
        with doc.create(subfigure(position='t', width=NoEscape(r'\linewidth'))):
            for i, n in enumerate(d_list):
                if i == 0:
                    with doc.create(subfigure(position='H', width=NoEscape(r'0.04\linewidth'))):
                        # string_append = r'\raisebox{0cm}{\rotatebox[origin=c]{90}{\scalebox{0.75}{}}}' + '%'
                        string_append = '\hfill'
                        doc.append(string_append)
                with doc.create(subfigure(position='H', width=NoEscape(r'0.32\linewidth'))):
                    name = f'$d_Z={n}$'
                    doc.append(Command('centering'))
                    doc.append(r'\rotatebox{0}{\scalebox{0.75}{%s}}' % name)
        counter = 0
        for idx, (i, j) in enumerate(itertools.product([1000, 5000, 10000], d_list)):
            if idx % len(d_list) == 0:
                name = f'$n={i}$'
                string_append = r'\raisebox{1.5cm}{\rotatebox[origin=c]{90}{\scalebox{0.75}{%s}}}' % name + '%\n'
            p = f'{dir}/figure_{j}_{i}.png'
            string_append += r'\includegraphics[width=0.32\linewidth]{%s}' % p + '%\n'
            counter += 1
            if counter == len(d_list):
                with doc.create(subfigure(position='H', width=NoEscape(r'\linewidth'))):
                    doc.append(string_append)
                counter = 0
    doc.generate_tex()

def break_plot(df,dir,mixed=True):
    if not os.path.exists(dir):
        os.makedirs(dir)
    methods = ['NCE_Q','real_weights','real_TRE_Q']

    for n in [1000,5000,10000]:
        for col_index,method in enumerate(methods):
            # for sep in [True,False]:
            # subset_3 = df[ (df['nce_style']==method) & (df['sep']==sep)  & (df['n']==n)].sort_values(['$/beta_{xz}$'])
            subset_3 = df[ (df['nce_style']==method)  & (df['n']==n)].sort_values(['$/beta_{xz}$'])
            a,b,e = calc_error_bars(subset_3['p_a=0.05'],alpha=0.05,num_samples=100)
                # appendix_string  = ' prod' if sep else ' mix'
                # format_string = dict_method[method] + appendix_string
            format_string = dict_method[method]
            plt.plot('$/beta_{xz}$', 'p_a=0.05', data=subset_3, linestyle='-', label=rf'{format_string}',
                     c=color_palette[col_index])

            # plt.fill_between(subset_3['beta_xy'], a, b, alpha=0.1)
            plt.fill_between(subset_3['$/beta_{xz}$'], a, b, alpha=0.1, color=color_palette[col_index])
            # plt.plot('$/beta_{xz}$','p_a=0.05',data=subset_3,linestyle='--', marker='o',label=rf'{format_string}')
            # plt.fill_between(subset_3['$/beta_{xz}$'], a, b, alpha=0.1)
        plt.hlines(0.05, 0, df['$/beta_{xz}$'].max())
        # plt.legend(prop={'size': 10})
        plt.xlabel(r'$\beta_{XZ}$')
        plt.ylabel('Size')
        plt.savefig(f'{dir}/figure_{n}.png',bbox_inches = 'tight',
    pad_inches = 0.05)
        plt.clf()
    for n in [1000,5000,10000]:
        for method in methods:
            # for sep in [True,False]:
            subset_3 = df[(df['nce_style'] == method) & (df['n'] == n)].sort_values(['$/beta_{xz}$'])
            a, b, e = calc_error_bars(subset_3['p_a=0.05'], alpha=0.05, num_samples=100)
            # appendix_string  = ' prod' if sep else ' mix'
            # format_string = dict_method[method] + appendix_string
            format_string = dict_method[method]
            plt.plot('$/beta_{xz}$', 'p_a=0.05', data=subset_3, linestyle='--', marker='o', label=rf'{format_string}')
            plt.fill_between(subset_3['$/beta_{xz}$'], a, b, alpha=0.1)
        plt.hlines(0.05, 0, 0.25)
        plt.legend(prop={'size': 10})
        plt.xlabel(r'ESS')
        plt.xticks(ticks=[0.0,0.05,0.1,0.15,0.20,0.25],labels=[6661,6647,6637,6621,6586,6539])
        plt.ylabel('Size')
        plt.savefig(f'{dir}/figure_{n}_ESS.png',bbox_inches = 'tight',
    pad_inches = 0.05)
        plt.clf()






def plot_14_hsic(ref_file,file,savedir):

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    df_1 = pd.read_csv(ref_file,index_col=0)
    df_1 = df_1[df_1['$/beta_{xz}$']==1.0]
    df_1_sub = df_1[df_1['beta_xy']==0.0].sort_values(['n'])

    df_2 = pd.read_csv(file, index_col=0)
    df_2 = df_2[df_2['$/beta_{xz}$']==1.0]

    df_2_sub = df_2[df_2['beta_xy'] == 0.0].sort_values(['n'])

    a, b, e = calc_error_bars(df_1_sub['p_a=0.05'], alpha=0.05, num_samples=100)
    plt.plot('n', 'p_a=0.05', data=df_1_sub, linestyle='--', marker='o', label=f'HSIC',c=color_palette[0])
    plt.fill_between(df_1_sub['n'], a, b, alpha=0.1,color=color_palette[0])

    a, b, e = calc_error_bars(df_2_sub['p_a=0.05'], alpha=0.05, num_samples=100)
    plt.plot('n', 'p_a=0.05', data=df_2_sub, linestyle='--', marker='o', label=f'KC-HSIC',c=color_palette[1])
    plt.fill_between(df_2_sub['n'], a, b, alpha=0.1,color=color_palette[1])

    plt.hlines(0.05, 1000, 10000)
    # plt.legend(prop={'size': 10})
    plt.xticks([1000,5000,10000])
    plt.xlabel(r'$n$')
    plt.ylabel(r'Size')
    plt.savefig(f'{savedir}/figure.png', bbox_inches='tight',
                pad_inches=0.05)
    plt.clf()

def plot_15_cond(ref_file,file,savedir):
    def calc_power(vec, level=.05):
        n = vec.shape[0]
        pow = np.sum(vec <= level) / n
        return pow

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    df_1 = pd.read_csv(ref_file,index_col=0)
    df_1_sub = df_1.sort_values(['n'])
    size = []
    for i in range(3):
        s = calc_power(df_1_sub.iloc[i,:].values)
        size.append(s)
    df_1_sub['p_a=0.05'] = size
    df_1_sub =df_1_sub.reset_index()
    df_2 = pd.read_csv(file, index_col=0)
    df_2_sub = df_2[df_2['beta_xy'] == 0.0].sort_values(['n'])



    a, b, e = calc_error_bars(df_1_sub['p_a=0.05'], alpha=0.05, num_samples=100)
    plt.plot('n', 'p_a=0.05', data=df_1_sub, linestyle='--', marker='o', label=f'partial copula',c=color_palette[0])
    plt.fill_between(df_1_sub['n'], a, b, alpha=0.1,color=color_palette[0])

    a, b, e = calc_error_bars(df_2_sub['p_a=0.05'], alpha=0.05, num_samples=100)
    plt.plot('n', 'p_a=0.05', data=df_2_sub, linestyle='--', marker='o', label=f'KC-HSIC',c=color_palette[1])
    plt.fill_between(df_2_sub['n'], a, b, alpha=0.1,color=color_palette[1])


    plt.hlines(0.05, 1000, 10000)
    # plt.legend(prop={'size': 10})
    plt.xticks([1000,5000,10000])
    plt.xlabel(r'$n$')
    plt.ylabel(r'Size')
    plt.savefig(f'{savedir}/figure.png', bbox_inches='tight',
                pad_inches=0.05)
    plt.clf()

if __name__ == '__main__':
    pass
    # # plot_2_true_weights('kc_rule_real_weights_2.csv','plot_2_real')
    # # plot_2_true_weights('do_null_mixed_real_new_2.csv','plot_3_real',True)
    # # plot_3_est_weights('do_null_mixed_est_3d_2.csv','mixed_est_3d_2',True)
    # # plot_15_cond('quantile_jmlr_break_ref.csv','old_csvs/cond_jobs_kc_real_rule.csv','plot_15')
    #
    """
    Cont business
    """
    df_1 = pd.read_csv('kc_rule_3_test_3d.csv',index_col=0)
    df_2 = pd.read_csv('kc_rule_3_test_3d_2.csv',index_col=0)
    df_3 = pd.read_csv('kc_rule_3_test_2.csv',index_col=0)
    # df_4 = pd.read_csv('cont_hdm_pow.csv',index_col=0)

    df = pd.concat([df_1,df_2,df_3],axis=0)
    df = df.groupby(['nce_style','d_Z','beta_xy','$/beta_{xz}$','n'])['p_a=0.05'].min().reset_index()
    # df = pd.concat([df,df_4],axis=0)
    plot_2_est_weights(df,'cont_plots_est')
    #
    """
    1d cont business linear
    # """
    df_1 = pd.read_csv("kc_rule_1d_linear_0.5_3.csv",index_col=0)
    df_1['nce_style'] = df_1['nce_style'].apply(lambda x: x+'_linear')
    # df_2 = pd.read_csv('kc_rule_3_test_2.csv',index_col=0)
    df_4 = pd.read_csv('cont_hdm_pow.csv',index_col=0)
    # df = pd.concat([df_1,df_2,df_4],axis=0)
    df = pd.concat([df_1,df_4],axis=0)
    df = df.groupby(['nce_style','d_Z','beta_xy','$/beta_{xz}$','n'])['p_a=0.05'].min().reset_index()
    plot_2_est_weights(df,'cont_plots_est_1D_linear')
    """
    1d cont business
    # """
    #
    df_1 = pd.read_csv('kc_rule_3_test_3d.csv', index_col=0)
    df_2 = pd.read_csv('kc_rule_3_test_3d_2.csv', index_col=0)
    df_3 = pd.read_csv('kc_rule_3_test_2.csv', index_col=0)
    df_4 = pd.read_csv('cont_hdm_pow.csv',index_col=0)
    df = pd.concat([df_1, df_2, df_3], axis=0)
    df = df.groupby(['nce_style', 'd_Z', 'beta_xy', '$/beta_{xz}$', 'n'])['p_a=0.05'].min().reset_index()
    df = pd.concat([df,df_4],axis=0)
    plot_2_est_weights(df, 'cont_plots_est_1D')
    #
    #
    #
    # #
    for y_index in [1,4]:
        hdm_break = pd.read_csv(f'hdm_breaker_fam_y={y_index}_job.csv',index_col=0)
        hdm_break = hdm_break.groupby(['nce_style','d_Z','beta_xy','$/beta_{xz}$','n'])['p_a=0.05'].min().reset_index()
        df_4 = pd.read_csv(f'break_hdm_pow_{y_index}.csv',index_col=0)
        hdm_break = pd.concat([hdm_break,df_4],axis=0)
        plot_2_est_weights(hdm_break,f'hdm_break_y={y_index}')

        hdm_break_1 = pd.read_csv(f'hdm_breaker_fam_y={y_index}_job_50.csv',index_col=0)
        hdm_break_1 = hdm_break_1.groupby(['nce_style','d_Z','beta_xy','$/beta_{xz}$','n'])['p_a=0.05'].min().reset_index()
        hdm_break_1 = hdm_break_1[hdm_break_1['beta_xy']==0.001]

        hdm_break_2 = pd.read_csv(f'hdm_breaker_fam_y={y_index}_job_50_2.csv',index_col=0)
        hdm_break_2 = hdm_break_2.groupby(['nce_style','d_Z','beta_xy','$/beta_{xz}$','n'])['p_a=0.05'].min().reset_index()
        # hdm_break = pd.concat([hdm_break,df_4],axis=0)
        hdm_break = pd.concat([hdm_break_1,hdm_break_2],axis=0)
        plot_2_est_weights(hdm_break,f'hdm_break_y={y_index}_50_2')
    #
    """
    MIXED PLOTS
    """
    mixed_df = pd.read_csv('do_null_mix_sanity_3_est.csv',index_col=0)
    bool_keep = (mixed_df['nce_style']=='real_weights') &  (mixed_df['d_Z']==50)
    mixed_df = mixed_df[~bool_keep]
    bool_keep = (mixed_df['nce_style']=='real_weights') &  (mixed_df['d_Z']==15)
    mixed_df = mixed_df[~bool_keep]
    fixed_reals = pd.read_csv('do_null_mix_real_test_0.05.csv',index_col=0)
    df = pd.concat([mixed_df,fixed_reals],axis=0)
    plot_3_est_weights(df,'mixed_sanity_3_est',mixed=True)
    """
    Linear break plot
    """
    df_1 = pd.read_csv('kc_rule_1d_linear_0.1.csv',index_col=0)
    df_2 = pd.read_csv('kc_rule_1d_linear_0.05.csv',index_col=0)
    df_3 = pd.read_csv('kc_rule_1d_linear_0.25.csv',index_col=0)
    df_4 = pd.read_csv('kc_rule_1d_linear_0.5.csv',index_col=0)
    df_5 = pd.read_csv('kc_rule_1d_linear_0.5_3.csv',index_col=0)
    df = pd.concat([df_1,df_2,df_3,df_4,df_5],axis=0)
    df = df[(df['beta_xy']==0.0)&(df['d_Z']==1)]
    break_plot(df,'linear_break')

    """
    General break plot
    """
    ablation_3d = pd.read_csv('ablation_3d.csv',index_col=0)

    break_plot(ablation_3d,'bd_hsic_breaker')

    """
    hsic and quantile plot
    """
    plot_14_hsic('old_csvs/ind_jobs_hsic_2.csv','old_csvs/hsic_break_real.csv','plot_14')
    plot_15_cond('rcit.csv','old_csvs/cond_jobs_kc_real_rule.csv','plot_15')
    plot_15_cond('rcot.csv','old_csvs/cond_jobs_kc_real_rule.csv','plot_15_b')
