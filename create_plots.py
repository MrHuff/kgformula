import pandas as pd

from post_processing_utils.plot_builder import *
from post_processing_utils.latex_plots import *
from pylatex import Document, Section, Figure, SubFigure, NoEscape,Command
from pylatex.base_classes import Environment
from pylatex.package import Package
import itertools

dict_method = {'NCE_Q': 'NCE-Q', 'real_TRE_Q': 'TRE-Q', 'random_uniform': 'random uniform', 'rulsif': 'RuLSIF'}

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

def plot_2_true_weights(csv,dir,mixed=False):
    df = pd.read_csv(csv, index_col=0)
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
    df = pd.read_csv(csv, index_col=0)
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
    df = pd.read_csv(csv, index_col=0)
    if not os.path.exists(dir):
        os.makedirs(dir)

    d_list = [1,3,15,50] if not mixed else [2,3,15,50]
    methods = df['nce_style'].unique().tolist()
    for d in d_list:
        subset = df[df['d_Z']==d].sort_values(['beta_xy'])
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
    df = pd.read_csv(csv, index_col=0)
    if not os.path.exists(dir):
        os.makedirs(dir)

    d_list = [1,3,15,50] if not mixed else [2,3,15,50]
    methods = df['nce_style'].unique().tolist()
    for d in d_list:
        subset = df[df['d_Z']==d].sort_values(['beta_xy'])
        for n in [1000, 5000, 10000]:
            subset_2 = subset[subset['n'] == n]
            for method in methods:
                for sep in [True,False]:
                    subset_3 = subset_2[ (subset_2['nce_style']==method) & (subset_2['sep']==sep) ]
                    a,b,e = calc_error_bars(subset_3['p_a=0.05'],alpha=0.05,num_samples=100)
                    appendix_string  = ' prod' if sep else ' mix'
                    format_string = dict_method[method] + appendix_string
                    plt.plot('beta_xy','p_a=0.05',data=subset_3,linestyle='--', marker='o',label=rf'{format_string}')
                    plt.fill_between(subset_3['beta_xy'], a, b, alpha=0.1)
            plt.hlines(0.05, 0, subset_2['beta_xy'].max())
            plt.legend(prop={'size': 10})
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

def break_plot(dir,mixed=True):
    df = pd.read_csv('old_csvs/do_null_mixed_est_new_break.csv', index_col=0)
    if not os.path.exists(dir):
        os.makedirs(dir)
    methods = ['NCE_Q','real_TRE_Q']

    for n in [1000,5000,10000]:
        for method in methods:
            for sep in [True,False]:
                subset_3 = df[ (df['nce_style']==method) & (df['sep']==sep)  & (df['n']==n)].sort_values(['$/beta_{xz}$'])
                a,b,e = calc_error_bars(subset_3['p_a=0.05'],alpha=0.05,num_samples=100)
                appendix_string  = ' prod' if sep else ' mix'
                format_string = dict_method[method] + appendix_string
                plt.plot('$/beta_{xz}$','p_a=0.05',data=subset_3,linestyle='--', marker='o',label=rf'{format_string}')
                plt.fill_between(subset_3['$/beta_{xz}$'], a, b, alpha=0.1)
        plt.hlines(0.05, 0, 0.25)
        plt.legend(prop={'size': 10})
        plt.xlabel(r'$\beta_{xz}$')
        plt.ylabel('Size')
        plt.savefig(f'{dir}/figure_{n}.png',bbox_inches = 'tight',
    pad_inches = 0.05)
        plt.clf()
    for n in [1000,5000,10000]:
        for method in methods:
            for sep in [True,False]:
                subset_3 = df[ (df['nce_style']==method) & (df['sep']==sep)  & (df['n']==n)].sort_values(['$/beta_{xz}$'])
                a,b,e = calc_error_bars(subset_3['p_a=0.05'],alpha=0.05,num_samples=100)
                appendix_string  = ' prod' if sep else ' mix'
                format_string = method.replace("_"," ") + appendix_string
                plt.plot('$/beta_{xz}$','p_a=0.05',data=subset_3,linestyle='--', marker='o',label=rf'{format_string}')
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
    plt.plot('n', 'p_a=0.05', data=df_1_sub, linestyle='--', marker='o', label=f'HSIC')
    plt.fill_between(df_1_sub['n'], a, b, alpha=0.1)

    a, b, e = calc_error_bars(df_2_sub['p_a=0.05'], alpha=0.05, num_samples=100)
    plt.plot('n', 'p_a=0.05', data=df_2_sub, linestyle='--', marker='o', label=f'KC-HSIC')
    plt.fill_between(df_2_sub['n'], a, b, alpha=0.1)

    plt.hlines(0.05, 1000, 10000)
    plt.legend(prop={'size': 10})
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
    plt.plot('n', 'p_a=0.05', data=df_1_sub, linestyle='--', marker='o', label=f'Partial Copula')
    plt.fill_between(df_1_sub['n'], a, b, alpha=0.1)

    a, b, e = calc_error_bars(df_2_sub['p_a=0.05'], alpha=0.05, num_samples=100)
    plt.plot('n', 'p_a=0.05', data=df_2_sub, linestyle='--', marker='o', label=f'KC-HSIC')
    plt.fill_between(df_2_sub['n'], a, b, alpha=0.1)

    plt.hlines(0.05, 1000, 10000)
    plt.legend(prop={'size': 10})
    plt.xticks([1000,5000,10000])
    plt.xlabel(r'$n$')
    plt.ylabel(r'Size')
    plt.savefig(f'{savedir}/figure.png', bbox_inches='tight',
                pad_inches=0.05)
    plt.clf()

def plot_power_ablation(path,dir):
    df = pd.read_csv(path, index_col=0)
    if not os.path.exists(dir):
        os.makedirs(dir)
    n = 10000
    methods = df['nce_style'].unique().tolist()
    df = df[(df['n'] == n) & (df['sep'] == True)].sort_values(['beta_xy'])

    for m in methods:
        subset_3 = df[(df['nce_style'] == m)].sort_values(
            ['beta_xy'])
        a, b, e = calc_error_bars(subset_3['p_a=0.05'], alpha=0.05, num_samples=100)
        appendix_string = ' prod'
        format_string = dict_method[m] + appendix_string
        plt.plot('beta_xy', 'p_a=0.05', data=subset_3, linestyle='--', marker='o', label=rf'{format_string}')
        plt.fill_between(subset_3['beta_xy'], a, b, alpha=0.1)
    plt.hlines(0.05, 0.01, 0.09)
    plt.legend(prop={'size': 10})
    plt.xticks( [0.01,0.03,0.05,0.07,0.09])
    plt.xlabel(r'$\beta_{XY}$')
    plt.ylabel(r'Power $\alpha=0.05$')
    plt.savefig(f'{dir}/figure_{n}.png', bbox_inches='tight',
                    pad_inches=0.05)
    plt.clf()

if __name__ == '__main__':
    pass
    # plot_2_true_weights('kc_rule_real_weights_2.csv','plot_2_real')
    # plot_2_true_weights('do_null_mixed_real_new_2.csv','plot_3_real',True)
    # plot_3_est_weights('do_null_mixed_est_new_2.csv','mixed_est',True)
    # plot_15_cond('quantile_jmlr_break_ref.csv','old_csvs/cond_jobs_kc_real_rule.csv','plot_15')

    plot_2_est_weights_test('kc_rule_3_test_3d.csv','plot_uniform_test_3d')
    # plot_2_true_weights('kc_rule_real_weights_3_test_2.csv','plot_real_test_2')


    # plot_2_true_weights('kc_rule_real_weights.csv','plot_2_real')
    # plot_2_true_weights('do_null_mixed_real_new.csv','plot_3_real',True)
    # plot_2_est_weights('kc_rule.csv','plot_4')
    # plot_3_est_weights('do_null_mixed_est_new.csv','mixed_est')
    # break_plot('break_kchsic')
    # plot_14_hsic('ind_jobs_hsic_2.csv','hsic_break_real.csv','plot_14')
    # plot_15_cond('quantile_jmlr_break_ref.csv','old_csvs/cond_jobs_kc_real_rule.csv','plot_15')
    # plot_power_ablation('ablation_mixed_2.csv','ablation_plot')