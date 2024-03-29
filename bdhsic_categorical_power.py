import os

import pandas as pd

from post_process import *
from scipy.stats import kstest
from create_plots import *

def plot_power(raw_df,dir,name,ests):
    for d in [1]:
        for n in [1000, 5000, 10000]:
            df = raw_df[raw_df['n'] == n]
            # for alp in [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]:
            for col_index,est in enumerate(ests):
                subset_3 = df[df['estimator'] == est].sort_values(['alp'])
                label_name = dict_method[est] if est!='HDM' else 'HDM'
                a,b,e = calc_error_bars(subset_3['alp=0.05'],alpha=0.05,num_samples=100)
                # plt.plot('alp','alp=0.05',data=subset_3,linestyle='--', marker='o',label=r'$n'+f'={n}$')
                # plt.plot('alp','alp=0.05',data=subset_3,linestyle='--', marker='o',label=f'{label_name}',c = plt.cm.rainbow(col_index/8.))
                plt.plot('alp','alp=0.05',data=subset_3,linestyle='-',label=f'{label_name}',c = col_dict[est])
                plt.fill_between(subset_3['alp'], a, b, alpha=0.1,color=col_dict[est])
            plt.hlines(0.05, 0, 0.1)
            # plt.legend(prop={'size': 10})
            plt.xticks(rotation=90)
            plt.xticks([0.0,0.005,0.01,2*1e-2,4*1e-2,6*1e-2,8*1e-2,1e-1])
            plt.xlabel(r'$\beta_{XY}$')
            plt.ylabel(r'Power $\alpha=0.05$')
            plt.savefig(f'{dir}/{name}_{n}.png',bbox_inches = 'tight',pad_inches = 0.05)
            plt.clf()

def calc_power(vec, level=.05):
    n = vec.shape[0]
    pow = np.sum(vec<=level)/n
    return pow
def transform_df(df):
    df['alp'] = df['beta_xy']
    df['null'] = False
    df['estimator'] = df['nce_style']
    df['alp=0.01'] = df['p_a=0.01']
    df['alp=0.05'] = df['p_a=0.05']
    df['alp=0.1'] = df['p_a=0.1']
    return df
def extract_properties(job_params):
    data_dir = job_params['job_dir']
    n = job_params['n']
    estimator = job_params['estimator']
    mode= job_params['mode']
    qdist = job_params['qdist']
    es = job_params['estimate']
    suffix = f'_qf=rule_qd={qdist}_m={mode}_s={0}_{100}_e={es}_est={estimator}_sp={es}_br={500}_n={n}'
    load_path = job_params['data_dir']+'/'+data_dir+'/'+f'p_val_array{suffix}.pt'
    string_base = job_params['data_dir'].split('_')
    alp = float(string_base[3].split('=')[-1])
    null = string_base[4].split('=')[-1]
    properties= [alp,null,n,estimator,data_dir]
    return properties,load_path,suffix

def post_process_jobs(bench_res_dir,job_csv):
    # bench_res_dir = '1d_cat_pow_kchsic'
    # job_dir = 'do_null_binary_all_1d'
    if not os.path.exists(bench_res_dir):
        os.makedirs(bench_res_dir)
    # print(benchmark_data)

    # jobs = os.listdir(job_dir)
    # df_dat = []
    # latex_plot_structure_1 = []
    # latex_plot_structure_2 = []
    # for j in jobs:
    #     job_params = load_obj(j, folder=f'{job_dir}/')
    #     props, load_path ,suffix = extract_properties(job_params)
    #     df_dat.append(props)
    #     p_vals = torch.load(load_path).cpu().numpy()
    #     if props[1] == 'False':
    #         for lvl in [0.01, 0.05, 0.1]:
    #             pow = calc_power(p_vals, lvl)
    #             props.append(pow)
        # else:
        #     if props[-1]==f'{job_dir}_layers=1_width=32_True':
        #         super_suff =  suffix + f'_{props[0]}'
        #         latex_plot_structure_1.append(props+[f'{bench_res_dir}/pval_{super_suff}.jpg'])
        #         # get_hist(p_vals, '/pval', bench_res_dir, super_suff, '', '', '', '', '')
        #     _, p_val_ks_test = kstest(p_vals, 'uniform')
        #     for i in range(3):
        #         props.append(p_val_ks_test)
        # df_dat.append(props)

    # df_latex_1 = pd.DataFrame(latex_plot_structure_1,columns=['alp','null','n','estimator','data_dir','path'])
    # df_latex_1 = df_latex_1[df_latex_1['alp'].isin([0.02,0.06,0.10]) & df_latex_1['estimator'].isin(['NCE_Q','real_weights'])].sort_values(['n','alp'])
    # doc = Document(default_filepath=bench_res_dir)
    # with doc.create(Figure(position='H')) as plot:
    #     with doc.create(subfigure(position='t', width=NoEscape(r'\linewidth'))):
    #         for i, n in enumerate([0.02,0.06,0.1]):
    #             if i == 0:
    #                 with doc.create(subfigure(position='H', width=NoEscape(r'0.04\linewidth'))):
    #                     # string_append = r'\raisebox{0cm}{\rotatebox[origin=c]{90}{\scalebox{0.75}{}}}' + '%'
    #                     string_append = '\hfill'
    #                     doc.append(string_append)
    #             with doc.create(subfigure(position='H', width=NoEscape(r'0.32\linewidth'))):
    #                 name = rf'$\alpha={n}$'
    #                 doc.append(Command('centering'))
    #                 doc.append(r'\rotatebox{0}{\scalebox{0.75}{%s}}' % name)
    #     counter = 0
    #     reset_col = 3
    #     for idx, (j, p) in enumerate(zip( df_latex_1['n'].tolist(), df_latex_1['path'].tolist())):
    #         if idx % reset_col == 0:
    #             name = f'$n={j}$'
    #
    #             string_append = r'\raisebox{1.5cm}{\rotatebox[origin=c]{90}{\scalebox{0.75}{%s}}}' % name + '%\n'
    #         string_append += r'\includegraphics[width=0.32\linewidth]{%s}' % p + '%\n'
    #         counter += 1
    #         if counter == reset_col:
    #             with doc.create(subfigure(position='H', width=NoEscape(r'\linewidth'))):
    #                 doc.append(string_append)
    #             counter = 0
    # doc.generate_tex()

    df_hdm = pd.read_csv('1d_cat_pow/pow_and_calib.csv')
    df_hdm['data_dir'] = None
    df_hdm['estimator'] = 'hdm'
    df_hdm = df_hdm[[ 'alp','null','n','estimator','alp=0.01','alp=0.05','alp=0.1']]
    df_old = pd.read_csv('do_null_binary_bench_old_statistic.csv')
    df_old = transform_df(df_old)
    df_old['estimator']='Singh et al.'
    df_cfme = pd.read_csv('do_null_binary_bench_cfme.csv')
    df_cfme = transform_df(df_cfme)
    df_cfme['estimator']='CfME'
    df = pd.read_csv(job_csv)
    df = transform_df(df)
    df = pd.concat([df,df_hdm,df_cfme,df_old],axis=0)
    # df = pd.concat([df,df_hdm],axis=0)
    # subset = df[(df['null']==False) | (df['null']=='False')]
    plot_power(df,bench_res_dir,'power_plot_sep',df['estimator'].unique().tolist())

if __name__ == '__main__':
    post_process_jobs('binary_perm_plot','do_null_binary_perm_cluster.csv')
    # post_process_jobs('binary_perm_plot_linear','do_null_binary_linear_kernel_perm_cluster.csv')
    # post_process_jobs('binary_perm_plot_cfme','do_null_binary_bench_cfme.csv')
    # post_process_jobs('binary_perm_plot_old_statistic','do_null_binary_bench_old_statistic.csv')



