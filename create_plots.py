from post_processing_utils.plot_builder import *
from post_processing_utils.latex_plots import *
from pylatex import Document, Section, Figure, SubFigure, NoEscape,Command
from pylatex.base_classes import Environment
from pylatex.package import Package
import itertools

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

def plot_1_true_weights():

    all_path = 'do_null_100/'
    small_path = 'base_jobs_kc_layers=3_width=32'
    df = pd.read_csv('base_jobs_kc.csv',index_col=0)
    df = df[df['beta_xy']==0.0]
    subset = df.loc[df.groupby(["n","d_Z"])["KS pval"].idxmax()].sort_values(['n','d_Z'])
    data_paths=[]
    suffix_paths=[]
    z_líst = []
    for index,row in subset.iterrows():
        dz = row['d_Z']
        suffix_paths.append(build_suffix(q_fac=row['$c_q$'],required_n=row['n'],estimator=row['nce_style'],br=500))
        data_paths.append(build_path(dx=dim_theta_phi[dz]['d_x'],
                                     dy=dim_theta_phi[dz]['d_y'],
                                     dz=dz,
                                     theta=dim_theta_phi[dz]['theta'],
                                     phi=dim_theta_phi[dz]['phi'],
                                     bxz=0.5,
                                     list_xy=[0,0.0],
                                     yz=[0.5,0.0])
                          )
        z_líst.append(dz)
    DIRNAME = 'plot_1_real'
    if os.path.exists(DIRNAME):
        shutil.rmtree(DIRNAME)
        os.makedirs(DIRNAME)
    else:
        os.makedirs(DIRNAME)
    plot_paths = []
    for i in range(len(data_paths)):
        full_file = f'{all_path}{data_paths[i]}{small_path}/pvalhsit_{suffix_paths[i]}.jpg'
        shutil.copy(full_file,DIRNAME+f'/pvalhsit_{suffix_paths[i]}_{z_líst[i]}.jpg')
        plot_paths.append(DIRNAME+f'/pvalhsit_{suffix_paths[i]}_{z_líst[i]}.jpg')

    doc = Document(default_filepath='plot_1_real')
    with doc.create(Figure(position='H')) as plot:
        with doc.create(subfigure(position='t', width=NoEscape(r'\linewidth'))):
            for i,n in enumerate([1,3,15,50]):
                if i==0:
                    with doc.create(subfigure(position='H', width=NoEscape(r'0.04\linewidth'))):
                        # string_append = r'\raisebox{0cm}{\rotatebox[origin=c]{90}{\scalebox{0.75}{}}}' + '%'
                        string_append = '\hfill'
                        doc.append(string_append)
                with doc.create(subfigure(position='H', width=NoEscape(r'0.24\linewidth'))):
                    name = f'$d_Z={n}$'
                    doc.append(Command('centering'))
                    doc.append(r'\rotatebox{0}{\scalebox{0.75}{%s}}'%name)
        counter=0
        for idx,(i, j, p) in enumerate(zip(subset['d_Z'].tolist(), subset['n'].tolist(), plot_paths)):
            if idx%4==0:
                name = f'$n={j}$'

                string_append=r'\raisebox{1.5cm}{\rotatebox[origin=c]{90}{\scalebox{0.75}{%s}}}'%name +'%\n'
            string_append+=r'\includegraphics[width=0.24\linewidth]{%s}'%p + '%\n'
            counter+=1
            if counter==4:
                with doc.create(subfigure(position='H', width=NoEscape(r'\linewidth'))):
                    doc.append(string_append)
                counter=0
    doc.generate_tex()

def plot_2_true_weights():
    df = pd.read_csv('base_jobs_kc.csv', index_col=0)
    df = df[df['beta_xy'] == 0.0]
    dir = 'plot_2_real'
    if not os.path.exists(dir):
        os.makedirs(dir)
    for d in [1,3,15,50]:
        subset = df[df['d_Z']==d].sort_values(['n'])
        for c in [0.2,0.4,0.6,0.8,1.0]:
            subset_2 = subset[subset['$c_q$']==c]
            a,b,e = calc_error_bars(subset_2['KS pval'],alpha=0.05,num_samples=100)
            plt.plot('n','KS pval',data=subset_2,linestyle='--', marker='o',label=f'$c_q={c}$')
            plt.fill_between(subset_2['n'], a, b, alpha=0.1)
        plt.hlines(0.05, 0, 10000)
        plt.legend(prop={'size': 10})
        plt.xlabel('$n$')
        plt.ylabel('p-val')
        plt.savefig(f'{dir}/figure_{d}.png',bbox_inches = 'tight',
    pad_inches = 0.05)
        plt.clf()
    doc = Document(default_filepath='plot_2_real')
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

def plot_3_true_weights():
    df = pd.read_csv('base_jobs_kc.csv', index_col=0)
    df = df[df['beta_xy'] != 0.0]
    dir = 'plot_3_real'
    if not os.path.exists(dir):
        os.makedirs(dir)
    for d in [1,3,15,50]:
        subset = df[df['d_Z']==d]
        for n in [1000,5000,10000]:
            subset_2 = subset[subset['n'] == n].sort_values(['beta_xy'])
            for c in [0.2,0.4,0.6,0.8,1.0]:
                subset_3 = subset_2[subset_2['$c_q$']==c]
                a,b,e = calc_error_bars(subset_3['p_a=0.05'],alpha=0.05,num_samples=100)
                plt.plot('beta_xy','p_a=0.05',data=subset_3,linestyle='--', marker='o',label=f'$c_q={c}$')
                plt.fill_between(subset_3['beta_xy'], a, b, alpha=0.1)
            plt.hlines(0.05, 0.1, 0.5)
            plt.legend(prop={'size': 10})
            plt.xticks([0.1,0.2,0.3,0.4,0.5])
            plt.xlabel(r'$\beta_{XY}$')
            plt.ylabel(r'Power $\alpha=0.05$')
            plt.savefig(f'{dir}/figure_{d}_{n}.png',bbox_inches = 'tight',
        pad_inches = 0.05)
            plt.clf()
    doc = Document(default_filepath='plot_3_real')
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
        for idx, (i, j) in enumerate(itertools.product([1000,5000,10000],[1,3,15,50])):
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


def plot_4_est_weights():

    all_path = 'do_null_100/'
    small_path = 'base_jobs_kc_est_layers=3_width=32'
    df = pd.read_csv('base_jobs_kc_est.csv',index_col=0)
    df = df[df['beta_xy']==0.0]
    subset = df.loc[df.groupby(["n","d_Z"])["KS pval"].idxmax()].sort_values(['n','d_Z'])
    data_paths=[]
    suffix_paths=[]
    z_list= []
    for index,row in subset.iterrows():
        dz = row['d_Z']
        suffix_paths.append(build_suffix(q_fac=row['$c_q$'],required_n=row['n'],estimator=row['nce_style'],br=500))
        data_paths.append(build_path(dx=dim_theta_phi[dz]['d_x'],
                                     dy=dim_theta_phi[dz]['d_y'],
                                     dz=dz,
                                     theta=dim_theta_phi[dz]['theta'],
                                     phi=dim_theta_phi[dz]['phi'],
                                     bxz=0.5,
                                     list_xy=[0,0.0],
                                     yz=[0.5,0.0])
                          )
        z_list.append(dz)
    DIRNAME = 'plot_4_est'
    if os.path.exists(DIRNAME):
        shutil.rmtree(DIRNAME)
        os.makedirs(DIRNAME)
    else:
        os.makedirs(DIRNAME)
    plot_paths = []
    for i in range(len(data_paths)):
        full_file = f'{all_path}{data_paths[i]}{small_path}/pvalhsit_{suffix_paths[i]}.jpg'
        shutil.copy(full_file,DIRNAME+f'/pvalhsit_{suffix_paths[i]}_{z_list[i]}.jpg')
        plot_paths.append(DIRNAME+f'/pvalhsit_{suffix_paths[i]}_{z_list[i]}.jpg')

    doc = Document(default_filepath=DIRNAME)
    with doc.create(Figure(position='H')) as plot:
        with doc.create(subfigure(position='t', width=NoEscape(r'\linewidth'))):
            for i,n in enumerate([1,3,15,50]):
                if i==0:
                    with doc.create(subfigure(position='H', width=NoEscape(r'0.04\linewidth'))):
                        # string_append = r'\raisebox{0cm}{\rotatebox[origin=c]{90}{\scalebox{0.75}{}}}' + '%'
                        string_append = '\hfill'
                        doc.append(string_append)
                with doc.create(subfigure(position='H', width=NoEscape(r'0.24\linewidth'))):
                    name = f'$d_Z={n}$'
                    doc.append(Command('centering'))
                    doc.append(r'\rotatebox{0}{\scalebox{0.75}{%s}}'%name)
        counter=0
        for idx,(i, j, p) in enumerate(zip(subset['d_Z'].tolist(), subset['n'].tolist(), plot_paths)):
            if idx%4==0:
                name = f'$n={j}$'

                string_append=r'\raisebox{1.5cm}{\rotatebox[origin=c]{90}{\scalebox{0.75}{%s}}}'%name +'%\n'
            string_append+=r'\includegraphics[width=0.24\linewidth]{%s}'%p + '%\n'
            counter+=1
            if counter==4:
                with doc.create(subfigure(position='H', width=NoEscape(r'\linewidth'))):
                    doc.append(string_append)
                counter=0
    doc.generate_tex()

def plot_5_est_weights():
    df_full = pd.read_csv('base_jobs_kc_est.csv', index_col=0)
    df_full = df_full[df_full['beta_xy'] == 0.0]
    dir = 'plot_5_est'
    if not os.path.exists(dir):
        os.makedirs(dir)
    for est in ['NCE_Q','real_TRE_Q']:
        df = df_full[df_full['nce_style']==est]
        for d in [1,3,15,50]:
            subset = df[df['d_Z']==d].sort_values(['n'])
            for c in [0.2,0.4,0.6,0.8,1.0]:
                subset_2 = subset[subset['$c_q$']==c]
                a,b,e = calc_error_bars(subset_2['KS pval'],alpha=0.05,num_samples=100)
                plt.plot('n','KS pval',data=subset_2,linestyle='--', marker='o',label=f'$c_q={c}$')
                plt.fill_between(subset_2['n'], a, b, alpha=0.1)
            plt.hlines(0.05, 0, 10000)
            plt.legend(prop={'size': 10})
            plt.xlabel('$n$')
            plt.ylabel('p-val')
            plt.savefig(f'{dir}/figure_{d}_{est}.png',bbox_inches = 'tight',
        pad_inches = 0.05)
            plt.clf()
    doc = Document(default_filepath=dir)
    with doc.create(Figure(position='H')) as plot:
        for est in ['NCE_Q', 'real_TRE_Q']:
            with doc.create(subfigure(position='t', width=NoEscape(r'\linewidth'))):
                with doc.create(subfigure(position='H', width=NoEscape(r'0.04\linewidth'))):
                    doc.append(Command('centering'))
                    string_append = r'\raisebox{0.8cm}{\rotatebox[origin=c]{90}{\scalebox{0.75}{%s}}}' % est
                    doc.append(string_append)
                for i, n in enumerate([1, 3, 15, 50]):
                    with doc.create(subfigure(position='H', width=NoEscape(r'0.24\linewidth'))):
                        name = f'$d_Z={n}$'
                        p = f'{dir}/figure_{n}_{est}.png'
                        doc.append(Command('centering'))
                        doc.append(r'\rotatebox{0}{\scalebox{0.75}{%s}}' % name)
                        doc.append(r'\includegraphics[width=\linewidth]{%s}'%p)
    doc.generate_tex()

def plot_6_est_weights():
    df = pd.read_csv('base_jobs_kc_est.csv', index_col=0)
    df = df[df['beta_xy'] != 0.0]
    df = df[df['nce_style']=='NCE_Q']
    dir = 'plot_6_est'
    if not os.path.exists(dir):
        os.makedirs(dir)
    for d in [1, 3, 15, 50]:
        subset = df[df['d_Z'] == d]
        for n in [1000, 5000, 10000]:
            subset_2 = subset[subset['n'] == n].sort_values(['beta_xy'])
            for c in [0.2, 0.4, 0.6, 0.8, 1.0]:
                subset_3 = subset_2[subset_2['$c_q$'] == c]
                a, b, e = calc_error_bars(subset_3['p_a=0.05'], alpha=0.05, num_samples=100)
                plt.plot('beta_xy', 'p_a=0.05', data=subset_3, linestyle='--', marker='o', label=f'$c_q={c}$')
                plt.fill_between(subset_3['beta_xy'], a, b, alpha=0.1)
            plt.hlines(0.05, 0.1, 0.5)
            plt.legend(prop={'size': 10})
            plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5])
            plt.xlabel(r'$\beta_{XY}$')
            plt.ylabel(r'Power $\alpha=0.05$')
            plt.savefig(f'{dir}/figure_{d}_{n}.png', bbox_inches='tight',
                        pad_inches=0.05)
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
def plot_7_est_weights():
    df = pd.read_csv('base_jobs_kc_est.csv', index_col=0)
    df = df[df['beta_xy'] != 0.0]
    df = df[df['nce_style']=='real_TRE_Q']
    dir = 'plot_7_est'
    if not os.path.exists(dir):
        os.makedirs(dir)
    for d in [1, 3, 15, 50]:
        subset = df[df['d_Z'] == d]
        for n in [1000, 5000, 10000]:
            subset_2 = subset[subset['n'] == n].sort_values(['beta_xy'])
            for c in [0.2, 0.4, 0.6, 0.8, 1.0]:
                subset_3 = subset_2[subset_2['$c_q$'] == c]
                a, b, e = calc_error_bars(subset_3['p_a=0.05'], alpha=0.05, num_samples=100)
                plt.plot('beta_xy', 'p_a=0.05', data=subset_3, linestyle='--', marker='o', label=f'$c_q={c}$')
                plt.fill_between(subset_3['beta_xy'], a, b, alpha=0.1)
            plt.hlines(0.05, 0.1, 0.5)
            plt.legend(prop={'size': 10})
            plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5])
            plt.xlabel(r'$\beta_{XY}$')
            plt.ylabel(r'Power $\alpha=0.05$')
            plt.savefig(f'{dir}/figure_{d}_{n}.png', bbox_inches='tight',
                        pad_inches=0.05)
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

def plot_8_rulsif():

    all_path = 'do_null_100/'
    small_path = 'base_jobs_kc_est_rulsif_layers=3_width=32'
    df = pd.read_csv('base_jobs_kc_est_rulsif.csv',index_col=0)
    df = df[df['beta_xy']==0.0]
    subset = df.loc[df.groupby(["n","d_Z"])["KS pval"].idxmax()].sort_values(['n','d_Z'])
    data_paths=[]
    suffix_paths=[]
    z_líst = []
    for index,row in subset.iterrows():
        dz = row['d_Z']
        suffix_paths.append(build_suffix(q_fac=row['$c_q$'],required_n=row['n'],estimator=row['nce_style'],br=500))
        data_paths.append(build_path(dx=dim_theta_phi[dz]['d_x'],
                                     dy=dim_theta_phi[dz]['d_y'],
                                     dz=dz,
                                     theta=dim_theta_phi[dz]['theta'],
                                     phi=dim_theta_phi[dz]['phi'],
                                     bxz=0.5,
                                     list_xy=[0,0.0],
                                     yz=[0.5,0.0])
                          )
        z_líst.append(dz)
    DIRNAME = 'plot_8_rulsif'
    if os.path.exists(DIRNAME):
        shutil.rmtree(DIRNAME)
        os.makedirs(DIRNAME)
    else:
        os.makedirs(DIRNAME)
    plot_paths = []
    for i in range(len(data_paths)):
        full_file = f'{all_path}{data_paths[i]}{small_path}/pvalhsit_{suffix_paths[i]}.jpg'
        shutil.copy(full_file,DIRNAME+f'/pvalhsit_{suffix_paths[i]}_{z_líst[i]}.jpg')
        plot_paths.append(DIRNAME+f'/pvalhsit_{suffix_paths[i]}_{z_líst[i]}.jpg')

    doc = Document(default_filepath=DIRNAME)
    with doc.create(Figure(position='H')) as plot:
        with doc.create(subfigure(position='t', width=NoEscape(r'\linewidth'))):
            for i,n in enumerate([1,3,15,50]):
                if i==0:
                    with doc.create(subfigure(position='H', width=NoEscape(r'0.04\linewidth'))):
                        # string_append = r'\raisebox{0cm}{\rotatebox[origin=c]{90}{\scalebox{0.75}{}}}' + '%'
                        string_append = '\hfill'
                        doc.append(string_append)
                with doc.create(subfigure(position='H', width=NoEscape(r'0.24\linewidth'))):
                    name = f'$d_Z={n}$'
                    doc.append(Command('centering'))
                    doc.append(r'\rotatebox{0}{\scalebox{0.75}{%s}}'%name)
        counter=0
        for idx,(i, j, p) in enumerate(zip(subset['d_Z'].tolist(), subset['n'].tolist(), plot_paths)):
            if idx%4==0:
                name = f'$n={j}$'

                string_append=r'\raisebox{1.5cm}{\rotatebox[origin=c]{90}{\scalebox{0.75}{%s}}}'%name +'%\n'
            string_append+=r'\includegraphics[width=0.24\linewidth]{%s}'%p + '%\n'
            counter+=1
            if counter==4:
                with doc.create(subfigure(position='H', width=NoEscape(r'\linewidth'))):
                    doc.append(string_append)
                counter=0
    doc.generate_tex()

def plot_9_rulsif():
    df_full = pd.read_csv('base_jobs_kc_est_rulsif.csv', index_col=0)
    df_full = df_full[df_full['beta_xy'] == 0.0]
    dir = 'plot_9_rulsif'
    if not os.path.exists(dir):
        os.makedirs(dir)
    df = df_full[df_full['nce_style']=='rulsif']
    for d in [1,3,15,50]:
        subset = df[df['d_Z']==d].sort_values(['n'])
        for c in [0.2,0.4,0.6,0.8,1.0]:
            subset_2 = subset[subset['$c_q$']==c]
            a,b,e = calc_error_bars(subset_2['KS pval'],alpha=0.05,num_samples=100)
            plt.plot('n','KS pval',data=subset_2,linestyle='--', marker='o',label=f'$c_q={c}$')
            plt.fill_between(subset_2['n'], a, b, alpha=0.1)
        plt.hlines(0.05, 0, 10000)
        plt.legend(prop={'size': 10})
        plt.xlabel('$n$')
        plt.ylabel('p-val')
        plt.savefig(f'{dir}/figure_{d}.png',bbox_inches = 'tight',
    pad_inches = 0.05)
        plt.clf()
    doc = Document(default_filepath=dir)
    with doc.create(Figure(position='H')) as plot:
        with doc.create(subfigure(position='t', width=NoEscape(r'\linewidth'))):
            for i, n in enumerate([1, 3, 15, 50]):
                with doc.create(subfigure(position='H', width=NoEscape(r'0.24\linewidth'))):
                    name = f'$d_Z={n}$'
                    p = f'{dir}/figure_{n}.png'
                    doc.append(Command('centering'))
                    doc.append(r'\rotatebox{0}{\scalebox{0.75}{%s}}' % name)
                    doc.append(r'\includegraphics[width=\linewidth]{%s}'%p)
    doc.generate_tex()


def plot_10_rulsif():
    df = pd.read_csv('base_jobs_kc_est_rulsif.csv', index_col=0)
    df = df[df['beta_xy'] != 0.0]
    df = df[df['nce_style']=='rulsif']
    dir = 'plot_10_rulsif'
    if not os.path.exists(dir):
        os.makedirs(dir)
    for d in [1, 3, 15, 50]:
        subset = df[df['d_Z'] == d]
        for n in [1000, 5000, 10000]:
            subset_2 = subset[subset['n'] == n].sort_values(['beta_xy'])
            for c in [0.2, 0.4, 0.6, 0.8, 1.0]:
                subset_3 = subset_2[subset_2['$c_q$'] == c]
                a, b, e = calc_error_bars(subset_3['p_a=0.05'], alpha=0.05, num_samples=100)
                plt.plot('beta_xy', 'p_a=0.05', data=subset_3, linestyle='--', marker='o', label=f'$c_q={c}$')
                plt.fill_between(subset_3['beta_xy'], a, b, alpha=0.1)
            plt.hlines(0.05, 0.1, 0.5)
            plt.legend(prop={'size': 10})
            plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5])
            plt.xlabel(r'$\beta_{XY}$')
            plt.ylabel(r'Power $\alpha=0.05$')
            plt.savefig(f'{dir}/figure_{d}_{n}.png', bbox_inches='tight',
                        pad_inches=0.05)
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

def plot_11_ablation():
    all_path = 'do_null_100/'
    small_path = 'base_jobs_kc_est_ablation_layers=3_width=32'
    df = pd.read_csv('base_jobs_kc_est_ablation.csv',index_col=0)
    df = df[df['beta_xy']==0.0]
    subset = df.loc[df.groupby(["n","d_Z"])["KS pval"].idxmax()].sort_values(['n','d_Z'])
    data_paths=[]
    suffix_paths=[]
    z_líst = []
    for index,row in subset.iterrows():
        dz = row['d_Z']
        suffix_paths.append(build_suffix(q_fac=row['$c_q$'],required_n=row['n'],estimator=row['nce_style'],br=500))
        data_paths.append(build_path(dx=dim_theta_phi[dz]['d_x'],
                                     dy=dim_theta_phi[dz]['d_y'],
                                     dz=dz,
                                     theta=dim_theta_phi[dz]['theta'],
                                     phi=dim_theta_phi[dz]['phi'],
                                     bxz=0.5,
                                     list_xy=[0,0.0],
                                     yz=[0.5,0.0])
                          )
        z_líst.append(dz)
    DIRNAME = 'plot_11_ablation'
    if os.path.exists(DIRNAME):
        shutil.rmtree(DIRNAME)
        os.makedirs(DIRNAME)
    else:
        os.makedirs(DIRNAME)
    plot_paths = []
    for i in range(len(data_paths)):
        full_file = f'{all_path}{data_paths[i]}{small_path}/pvalhsit_{suffix_paths[i]}.jpg'
        shutil.copy(full_file,DIRNAME+f'/pvalhsit_{suffix_paths[i]}_{z_líst[i]}.jpg')
        plot_paths.append(DIRNAME+f'/pvalhsit_{suffix_paths[i]}_{z_líst[i]}.jpg')

    doc = Document(default_filepath=DIRNAME)
    with doc.create(Figure(position='H')) as plot:
        with doc.create(subfigure(position='t', width=NoEscape(r'\linewidth'))):
            for i,n in enumerate([1,3,15,50]):
                if i==0:
                    with doc.create(subfigure(position='H', width=NoEscape(r'0.04\linewidth'))):
                        # string_append = r'\raisebox{0cm}{\rotatebox[origin=c]{90}{\scalebox{0.75}{}}}' + '%'
                        string_append = '\hfill'
                        doc.append(string_append)
                with doc.create(subfigure(position='H', width=NoEscape(r'0.24\linewidth'))):
                    name = f'$d_Z={n}$'
                    doc.append(Command('centering'))
                    doc.append(r'\rotatebox{0}{\scalebox{0.75}{%s}}'%name)
        counter=0
        for idx,(i, j, p) in enumerate(zip(subset['d_Z'].tolist(), subset['n'].tolist(), plot_paths)):
            if idx%4==0:
                name = f'$n={j}$'

                string_append=r'\raisebox{1.5cm}{\rotatebox[origin=c]{90}{\scalebox{0.75}{%s}}}'%name +'%\n'
            string_append+=r'\includegraphics[width=0.24\linewidth]{%s}'%p + '%\n'
            counter+=1
            if counter==4:
                with doc.create(subfigure(position='H', width=NoEscape(r'\linewidth'))):
                    doc.append(string_append)
                counter=0
    doc.generate_tex()

def plot_12_ablation():
    df_full = pd.read_csv('base_jobs_kc_est_ablation.csv', index_col=0)
    df_full = df_full[df_full['beta_xy'] == 0.0]
    dir = 'plot_12_ablation'
    if not os.path.exists(dir):
        os.makedirs(dir)
    for est in ['NCE_Q']:
        df = df_full[df_full['nce_style']==est]
        for d in [1,3,15,50]:
            subset = df[df['d_Z']==d].sort_values(['n'])
            for c in [0.2,0.4,0.6,0.8,1.0]:
                subset_2 = subset[subset['$c_q$']==c]
                a,b,e = calc_error_bars(subset_2['KS pval'],alpha=0.05,num_samples=100)
                plt.plot('n','KS pval',data=subset_2,linestyle='--', marker='o',label=f'$c_q={c}$')
                plt.fill_between(subset_2['n'], a, b, alpha=0.1)
            plt.hlines(0.05, 0, 10000)
            plt.legend(prop={'size': 10})
            plt.xlabel('$n$')
            plt.ylabel('p-val')
            plt.savefig(f'{dir}/figure_{d}_{est}.png',bbox_inches = 'tight',
        pad_inches = 0.05)
            plt.clf()
    doc = Document(default_filepath=dir)
    with doc.create(Figure(position='H')) as plot:
        for est in ['NCE_Q']:
            with doc.create(subfigure(position='t', width=NoEscape(r'\linewidth'))):
                with doc.create(subfigure(position='H', width=NoEscape(r'0.04\linewidth'))):
                    doc.append(Command('centering'))
                    string_append = r'\raisebox{0.8cm}{\rotatebox[origin=c]{90}{\scalebox{0.75}{%s}}}' % est
                    doc.append(string_append)
                for i, n in enumerate([1, 3, 15, 50]):
                    with doc.create(subfigure(position='H', width=NoEscape(r'0.24\linewidth'))):
                        name = f'$d_Z={n}$'
                        p = f'{dir}/figure_{n}_{est}.png'
                        doc.append(Command('centering'))
                        doc.append(r'\rotatebox{0}{\scalebox{0.75}{%s}}' % name)
                        doc.append(r'\includegraphics[width=\linewidth]{%s}'%p)
    doc.generate_tex()
def plot_13_ablation():
    df = pd.read_csv('base_jobs_kc_est_ablation.csv', index_col=0)
    df = df[df['beta_xy'] != 0.0]
    df = df[df['nce_style']=='NCE_Q']
    dir = 'plot_13_ablation'
    if not os.path.exists(dir):
        os.makedirs(dir)
    for d in [1, 3, 15, 50]:
        subset = df[df['d_Z'] == d]
        for n in [1000, 5000, 10000]:
            subset_2 = subset[subset['n'] == n].sort_values(['beta_xy'])
            for c in [0.2, 0.4, 0.6, 0.8, 1.0]:
                subset_3 = subset_2[subset_2['$c_q$'] == c]
                a, b, e = calc_error_bars(subset_3['p_a=0.05'], alpha=0.05, num_samples=100)
                plt.plot('beta_xy', 'p_a=0.05', data=subset_3, linestyle='--', marker='o', label=f'$c_q={c}$')
                plt.fill_between(subset_3['beta_xy'], a, b, alpha=0.1)
            plt.hlines(0.05, 0.1, 0.5)
            plt.legend(prop={'size': 10})
            plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5])
            plt.xlabel(r'$\beta_{XY}$')
            plt.ylabel(r'Power $\alpha=0.05$')
            plt.savefig(f'{dir}/figure_{d}_{n}.png', bbox_inches='tight',
                        pad_inches=0.05)
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

def plot_14_hsic():

    all_path = 'exp_hsic_break_100/'
    small_path = 'ind_jobs_hsic'
    df = pd.read_csv('ind_jobs_hsic.csv',index_col=0)
    df = df[df['beta_xy']==0.0]
    subset = df.loc[df.groupby(["n","d_Z"])["KS pval"].idxmax()].sort_values(['n','d_Z'])
    data_paths=[]
    suffix_paths=[]
    z_list = []
    for index,row in subset.iterrows():
        dz = int(row['d_Z'])
        suffix_paths.append(build_hsic(required_n=int(row['n']),br=500))
        data_paths.append(build_path(dx=dim_theta_phi_hsic[dz]['d_x'],
                                     dy=dim_theta_phi_hsic[dz]['d_y'],
                                     dz=dz,
                                     theta=dim_theta_phi_hsic[dz]['theta'],
                                     phi=dim_theta_phi_hsic[dz]['phi'],
                                     bxz=0.5,
                                     list_xy=[0.0,0.0],
                                     yz=[0.5,0.0])
                          )
        z_list.append(dz)
    DIRNAME = 'plot_14_hsic'
    if os.path.exists(DIRNAME):
        shutil.rmtree(DIRNAME)
        os.makedirs(DIRNAME)
    else:
        os.makedirs(DIRNAME)
    plot_paths = []
    for i in range(len(data_paths)):
        full_file = f'{all_path}{data_paths[i]}{small_path}/pvalhsit_{suffix_paths[i]}.jpg'
        shutil.copy(full_file,DIRNAME+f'/pvalhsit_{suffix_paths[i]}_{z_list[i]}.jpg')
        plot_paths.append(DIRNAME+f'/pvalhsit_{suffix_paths[i]}_{z_list[i]}.jpg')

    doc = Document(default_filepath=DIRNAME)
    with doc.create(Figure(position='H')) as plot:
        with doc.create(subfigure(position='t', width=NoEscape(r'\linewidth'))):
            for i,n in enumerate([1000,5000,10000]):
                if i==0:
                    with doc.create(subfigure(position='H', width=NoEscape(r'0.04\linewidth'))):
                        string_append = '\hfill'
                        doc.append(string_append)
                with doc.create(subfigure(position='H', width=NoEscape(r'0.32\linewidth'))):
                    name = f'$n={n}$'
                    doc.append(Command('centering'))
                    doc.append(r'\rotatebox{0}{\scalebox{0.75}{%s}}'%name)
        counter=0
        for idx,(i, j, p) in enumerate(zip(subset['d_Z'].tolist(), subset['n'].tolist(), plot_paths)):
            if idx%4==0:
                name = 'HSIC'
                string_append=r'\raisebox{1.5cm}{\rotatebox[origin=c]{90}{\scalebox{0.75}{%s}}}'%name +'%\n'
            string_append+=r'\includegraphics[width=0.32\linewidth]{%s}'%p + '%\n'
            counter+=1
            if counter==3:
                with doc.create(subfigure(position='H', width=NoEscape(r'\linewidth'))):
                    doc.append(string_append)
                counter=0
    doc.generate_tex()

def plot_15_hsic():

    all_path = 'exp_hsic_break_100/'
    small_path = 'hsic_jobs_kc_layers=3_width=32'
    df = pd.read_csv('hsic_jobs_kc.csv',index_col=0)
    df = df[df['beta_xy']==0.0]
    subset = df.loc[df.groupby(["n","d_Z"])["KS pval"].idxmax()].sort_values(['n','d_Z'])
    data_paths = []
    suffix_paths = []
    z_list = []
    for index, row in subset.iterrows():
        dz = row['d_Z']
        suffix_paths.append(build_suffix(q_fac=row['$c_q$'], required_n=row['n'], estimator=row['nce_style'], br=500))
        data_paths.append(build_path(dx=dim_theta_phi_hsic[dz]['d_x'],
                                     dy=dim_theta_phi_hsic[dz]['d_y'],
                                     dz=dz,
                                     theta=dim_theta_phi_hsic[dz]['theta'],
                                     phi=dim_theta_phi_hsic[dz]['phi'],
                                     bxz=0.5,
                                     list_xy=[0.0, 0.0],
                                     yz=[0.5, 0.0])
                          )
        z_list.append(dz)
    DIRNAME = 'plot_15_hsic'
    if os.path.exists(DIRNAME):
        shutil.rmtree(DIRNAME)
        os.makedirs(DIRNAME)
    else:
        os.makedirs(DIRNAME)
    plot_paths = []
    for i in range(len(data_paths)):
        full_file = f'{all_path}{data_paths[i]}{small_path}/pvalhsit_{suffix_paths[i]}.jpg'
        shutil.copy(full_file,DIRNAME+f'/pvalhsit_{suffix_paths[i]}_{z_list[i]}.jpg')
        plot_paths.append(DIRNAME+f'/pvalhsit_{suffix_paths[i]}_{z_list[i]}.jpg')

    doc = Document(default_filepath=DIRNAME)
    with doc.create(Figure(position='H')) as plot:
        with doc.create(subfigure(position='t', width=NoEscape(r'\linewidth'))):
            for i,n in enumerate([1000,5000,10000]):
                if i==0:
                    with doc.create(subfigure(position='H', width=NoEscape(r'0.04\linewidth'))):
                        string_append = '\hfill'
                        doc.append(string_append)
                with doc.create(subfigure(position='H', width=NoEscape(r'0.32\linewidth'))):
                    name = f'$n={n}$'
                    doc.append(Command('centering'))
                    doc.append(r'\rotatebox{0}{\scalebox{0.75}{%s}}'%name)
        counter=0
        for idx,(i, j, p) in enumerate(zip(subset['d_Z'].tolist(), subset['n'].tolist(), plot_paths)):
            if idx%4==0:
                name = 'KC-HSIC'
                string_append=r'\raisebox{1.5cm}{\rotatebox[origin=c]{90}{\scalebox{0.75}{%s}}}'%name +'%\n'
            string_append+=r'\includegraphics[width=0.32\linewidth]{%s}'%p + '%\n'
            counter+=1
            if counter==3:
                with doc.create(subfigure(position='H', width=NoEscape(r'\linewidth'))):
                    doc.append(string_append)
                counter=0
    doc.generate_tex()

def plot_16_hsic():
    df = pd.read_csv('hsic_jobs_kc.csv', index_col=0)
    df = df[df['beta_xy'] == 0.0]
    dir = 'plot_16_hsic'
    if not os.path.exists(dir):
        os.makedirs(dir)
    for d in [1]:
        subset = df[df['d_Z']==d].sort_values(['n'])
        for c in [0.2,0.4,0.6,0.8,1.0]:
            subset_2 = subset[subset['$c_q$']==c]
            a,b,e = calc_error_bars(subset_2['KS pval'],alpha=0.05,num_samples=100)
            plt.plot('n','KS pval',data=subset_2,linestyle='--', marker='o',label=f'$c_q={c}$')
            plt.fill_between(subset_2['n'], a, b, alpha=0.1)
        plt.hlines(0.05, 0, 10000)
        plt.legend(prop={'size': 10})
        plt.xlabel('$n$')
        plt.ylabel('p-val')
        plt.savefig(f'{dir}/figure_{d}.png',bbox_inches = 'tight',
    pad_inches = 0.05)
        plt.clf()
    doc = Document(default_filepath=dir)
    with doc.create(Figure(position='H')) as plot:
        with doc.create(subfigure(position='t', width=NoEscape(r'\linewidth'))):
            for i, n in enumerate([1]):
                with doc.create(subfigure(position='H', width=NoEscape(r'0.5\linewidth'))):
                    name = f'KC-HSIC'
                    p = f'{dir}/figure_{n}.png'
                    doc.append(Command('centering'))
                    doc.append(r'\rotatebox{0}{\scalebox{0.75}{%s}}' % name)
                    doc.append(r'\includegraphics[width=\linewidth]{%s}'%p)

    doc.generate_tex()

def plot_17_hsic():
    df = pd.read_csv('ind_jobs_hsic.csv', index_col=0)
    df = df[df['beta_xy'] == 0.0]
    dir = 'plot_17_hsic'
    if not os.path.exists(dir):
        os.makedirs(dir)
    for d in [1]:
        subset_2 = df[df['d_Z']==d].sort_values(['n'])
        a,b,e = calc_error_bars(subset_2['KS pval'],alpha=0.05,num_samples=100)
        plt.plot('n','KS pval',data=subset_2,linestyle='--', marker='o')
        plt.fill_between(subset_2['n'], a, b, alpha=0.1)
        plt.hlines(0.05, 0, 10000)
        plt.xlabel('$n$')
        plt.ylabel('p-val')
        plt.savefig(f'{dir}/figure_{d}.png',bbox_inches = 'tight',
    pad_inches = 0.05)
        plt.clf()
    doc = Document(default_filepath=dir)
    with doc.create(Figure(position='H')) as plot:
        with doc.create(subfigure(position='t', width=NoEscape(r'\linewidth'))):
            for i, n in enumerate([1]):
                with doc.create(subfigure(position='H', width=NoEscape(r'0.5\linewidth'))):
                    name = f'HSIC'
                    p = f'{dir}/figure_{n}.png'
                    doc.append(Command('centering'))
                    doc.append(r'\rotatebox{0}{\scalebox{0.75}{%s}}' % name)
                    doc.append(r'\includegraphics[width=\linewidth]{%s}'%p)

    doc.generate_tex()


def plot_18_gcm():
    all_path = 'exp_gcm_break_100/'
    small_path = 'cond_jobs_regression'
    df = pd.read_csv('cond_jobs_regression.csv',index_col=0)
    df = df[df['beta_xy']==0.0]
    subset = df.loc[df.groupby(["n","d_Z"])["KS pval"].idxmax()].sort_values(['n','d_Z'])
    data_paths=[]
    suffix_paths=[]
    z_list = []
    for index,row in subset.iterrows():
        dz = int(row['d_Z'])
        suffix_paths.append(build_regression(required_n=int(row['n'])))
        data_paths.append(build_path(dx=dim_theta_phi_gcm[dz]['d_x'],
                                     dy=dim_theta_phi_gcm[dz]['d_y'],
                                     dz=dz,
                                     theta=dim_theta_phi_gcm[dz]['theta'],
                                     phi=dim_theta_phi_gcm[dz]['phi'],
                                     bxz=0.0,
                                     list_xy=[0.0,0.0],
                                     yz=[-0.5,4.0])
                          )
        z_list.append(dz)
    DIRNAME = 'plot_18_gcm'
    if os.path.exists(DIRNAME):
        shutil.rmtree(DIRNAME)
        os.makedirs(DIRNAME)
    else:
        os.makedirs(DIRNAME)
    plot_paths = []
    for i in range(len(data_paths)):
        full_file = f'{all_path}{data_paths[i]}{small_path}/pvalhsit_{suffix_paths[i]}.jpg'
        shutil.copy(full_file,DIRNAME+f'/pvalhsit_{suffix_paths[i]}_{z_list[i]}.jpg')
        plot_paths.append(DIRNAME+f'/pvalhsit_{suffix_paths[i]}_{z_list[i]}.jpg')
    doc = Document(default_filepath=DIRNAME)
    with doc.create(Figure(position='H')) as plot:
        with doc.create(subfigure(position='t', width=NoEscape(r'\linewidth'))):
            for i,n in enumerate([1000,5000,10000]):
                if i==0:
                    with doc.create(subfigure(position='H', width=NoEscape(r'0.04\linewidth'))):
                        string_append = '\hfill'
                        doc.append(string_append)
                with doc.create(subfigure(position='H', width=NoEscape(r'0.32\linewidth'))):
                    name = f'$n={n}$'
                    doc.append(Command('centering'))
                    doc.append(r'\rotatebox{0}{\scalebox{0.75}{%s}}'%name)
        counter=0
        for idx,(i, j, p) in enumerate(zip(subset['d_Z'].tolist(), subset['n'].tolist(), plot_paths)):
            if idx%4==0:
                name = 'HSIC'
                string_append=r'\raisebox{1.5cm}{\rotatebox[origin=c]{90}{\scalebox{0.75}{%s}}}'%name +'%\n'
            string_append+=r'\includegraphics[width=0.32\linewidth]{%s}'%p + '%\n'
            counter+=1
            if counter==3:
                with doc.create(subfigure(position='H', width=NoEscape(r'\linewidth'))):
                    doc.append(string_append)
                counter=0
    doc.generate_tex()

def plot_19_gcm():

    all_path = 'exp_gcm_break_100/'
    small_path = 'cond_jobs_kc_layers=3_width=32'
    df = pd.read_csv('cond_jobs_kc.csv',index_col=0)
    df = df[df['beta_xy']==0.0]
    subset = df.loc[df.groupby(["n","d_Z"])["KS pval"].idxmax()].sort_values(['n','d_Z'])
    data_paths = []
    suffix_paths = []
    z_list=[]
    for index, row in subset.iterrows():
        dz = row['d_Z']
        suffix_paths.append(build_suffix(q_fac=row['$c_q$'], required_n=row['n'], estimator=row['nce_style'], br=500))
        data_paths.append(build_path(dx=dim_theta_phi_gcm[dz]['d_x'],
                                     dy=dim_theta_phi_gcm[dz]['d_y'],
                                     dz=dz,
                                     theta=dim_theta_phi_gcm[dz]['theta'],
                                     phi=dim_theta_phi_gcm[dz]['phi'],
                                     bxz=0.0,
                                     list_xy=[0.0,0.0],
                                     yz=[-0.5,4.0])
                          )
        z_list.append(dz)
    DIRNAME = 'plot_19_gcm'
    if os.path.exists(DIRNAME):
        shutil.rmtree(DIRNAME)
        os.makedirs(DIRNAME)
    else:
        os.makedirs(DIRNAME)
    plot_paths = []
    for i in range(len(data_paths)):
        full_file = f'{all_path}{data_paths[i]}{small_path}/pvalhsit_{suffix_paths[i]}.jpg'
        shutil.copy(full_file,DIRNAME+f'/pvalhsit_{suffix_paths[i]}_{z_list[i]}.jpg')
        plot_paths.append(DIRNAME+f'/pvalhsit_{suffix_paths[i]}_{z_list[i]}.jpg')

    doc = Document(default_filepath=DIRNAME)
    with doc.create(Figure(position='H')) as plot:
        with doc.create(subfigure(position='t', width=NoEscape(r'\linewidth'))):
            for i,n in enumerate([1000,5000,10000]):
                if i==0:
                    with doc.create(subfigure(position='H', width=NoEscape(r'0.04\linewidth'))):
                        string_append = '\hfill'
                        doc.append(string_append)
                with doc.create(subfigure(position='H', width=NoEscape(r'0.32\linewidth'))):
                    name = f'$n={n}$'
                    doc.append(Command('centering'))
                    doc.append(r'\rotatebox{0}{\scalebox{0.75}{%s}}'%name)
        counter=0
        for idx,(i, j, p) in enumerate(zip(subset['d_Z'].tolist(), subset['n'].tolist(), plot_paths)):
            if idx%4==0:
                name = 'KC-HSIC'
                string_append=r'\raisebox{1.5cm}{\rotatebox[origin=c]{90}{\scalebox{0.75}{%s}}}'%name +'%\n'
            string_append+=r'\includegraphics[width=0.32\linewidth]{%s}'%p + '%\n'
            counter+=1
            if counter==3:
                with doc.create(subfigure(position='H', width=NoEscape(r'\linewidth'))):
                    doc.append(string_append)
                counter=0
    doc.generate_tex()

def plot_21_gcm():
    df = pd.read_csv('cond_jobs_kc.csv', index_col=0)
    df = df[df['beta_xy'] == 0.0]
    dir = 'plot_21_gcm'
    if not os.path.exists(dir):
        os.makedirs(dir)
    for d in [1]:
        subset = df[df['d_Z']==d].sort_values(['n'])
        for c in [0.2,0.4,0.6,0.8,1.0]:
            subset_2 = subset[subset['$c_q$']==c]
            a,b,e = calc_error_bars(subset_2['KS pval'],alpha=0.05,num_samples=100)
            plt.plot('n','KS pval',data=subset_2,linestyle='--', marker='o',label=f'$c_q={c}$')
            plt.fill_between(subset_2['n'], a, b, alpha=0.1)
        plt.hlines(0.05, 0, 10000)
        plt.legend(prop={'size': 10})
        plt.xlabel('$n$')
        plt.ylabel('p-val')
        plt.savefig(f'{dir}/figure_{d}.png',bbox_inches = 'tight',
    pad_inches = 0.05)
        plt.clf()
    doc = Document(default_filepath=dir)
    with doc.create(Figure(position='H')) as plot:
        with doc.create(subfigure(position='t', width=NoEscape(r'\linewidth'))):
            for i, n in enumerate([1]):
                with doc.create(subfigure(position='H', width=NoEscape(r'0.5\linewidth'))):
                    name = f'KC-HSIC'
                    p = f'{dir}/figure_{n}.png'
                    doc.append(Command('centering'))
                    doc.append(r'\rotatebox{0}{\scalebox{0.75}{%s}}' % name)
                    doc.append(r'\includegraphics[width=\linewidth]{%s}'%p)

    doc.generate_tex()

def plot_20_gcm():
    df = pd.read_csv('cond_jobs_regression.csv', index_col=0)
    df = df[df['beta_xy'] == 0.0]
    dir = 'plot_20_gcm'
    if not os.path.exists(dir):
        os.makedirs(dir)
    for d in [1]:
        subset_2 = df[df['d_Z']==d].sort_values(['n'])
        a,b,e = calc_error_bars(subset_2['KS pval'],alpha=0.05,num_samples=100)
        plt.plot('n','KS pval',data=subset_2,linestyle='--', marker='o')
        plt.fill_between(subset_2['n'], a, b, alpha=0.1)
        plt.hlines(0.05, 0, 10000)
        plt.xlabel('$n$')
        plt.ylabel('p-val')
        plt.savefig(f'{dir}/figure_{d}.png',bbox_inches = 'tight',
    pad_inches = 0.05)
        plt.clf()
    doc = Document(default_filepath=dir)
    with doc.create(Figure(position='H')) as plot:
        with doc.create(subfigure(position='t', width=NoEscape(r'\linewidth'))):
            for i, n in enumerate([1]):
                with doc.create(subfigure(position='H', width=NoEscape(r'0.5\linewidth'))):
                    name = f'HSIC'
                    p = f'{dir}/figure_{n}.png'
                    doc.append(Command('centering'))
                    doc.append(r'\rotatebox{0}{\scalebox{0.75}{%s}}' % name)
                    doc.append(r'\includegraphics[width=\linewidth]{%s}'%p)

    doc.generate_tex()
if __name__ == '__main__':
    pass
    # plot_1_true_weights()
    # plot_2_true_weights()
    # plot_3_true_weights()
    # plot_4_est_weights()
    # plot_5_est_weights()
    # plot_6_est_weights()
    # plot_7_est_weights()
    # plot_8_rulsif()
    # plot_9_rulsif()
    # plot_10_rulsif()
    # plot_11_ablation()
    # plot_12_ablation()
    # plot_13_ablation()
    # plot_14_hsic()
    # plot_15_hsic()
    # plot_17_hsic()
    # plot_18_gcm()
    # plot_19_gcm()
    # plot_20_gcm()
    # plot_21_gcm()


