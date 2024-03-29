from scipy.stats import kstest
from create_plots import *


def plot_power(raw_df,dir,name):
    for d in [1]:
        df = raw_df.sort_values(['alp'])
        # for alp in [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]:
        for n in [1000,5000,10000]:
            subset_3 = df[df['n']==n]
            a,b,e = calc_error_bars(subset_3['alp=0.05'],alpha=0.05,num_samples=100)
            plt.plot('alp','alp=0.05',data=subset_3,linestyle='--', marker='o',label=r'$n'+f'={n}$')
            plt.fill_between(subset_3['alp'], a, b, alpha=0.1)
        plt.hlines(0.05, 0, 0.1)
        plt.legend(prop={'size': 10})
        plt.xticks(rotation=90)
        plt.xticks([0.0,0.005,0.01,2*1e-2,4*1e-2,6*1e-2,8*1e-2,1e-1])
        plt.xlabel(r'$\beta_{XY}$')
        plt.ylabel(r'Power $\alpha=0.05$')
        plt.savefig(f'{dir}/{name}_hde.png',bbox_inches = 'tight',pad_inches = 0.05)
        plt.clf()


def calc_power(vec, level=.05):
    n = vec.shape[0]
    pow = np.sum(vec<=level)/n
    return pow


bench_res_dir = '1d_cont_pow'
benchmark_data = pd.read_csv('hdm_bench_syntehtic_cont.csv')
if not os.path.exists(bench_res_dir):
    os.makedirs(bench_res_dir)
# print(benchmark_data)
bench_extract_cols = list(str(el) for el in range(1,101))


pow_and_calib = []
for i,row in benchmark_data.iterrows():
    data_row = []
    p_vals = row[bench_extract_cols].values.astype(float).squeeze()
    for lvl in [0.001,0.01,0.025,0.05,0.1]:
        pow = calc_power(p_vals,lvl)
        data_row.append(pow)
    stat,p_val_ks_test = kstest(p_vals,'uniform')
    data_row.append(stat)
    data_row.append(p_val_ks_test)
    pow_and_calib.append(data_row)

df_calib_pow = pd.DataFrame(pow_and_calib,columns=['p_a=0.001','p_a=0.01' ,'p_a=0.025' , 'p_a=0.05' ,'p_a=0.1','KS stat','KS pval'])
big_df = pd.concat([benchmark_data,df_calib_pow],axis=1)
big_df.to_csv(f"{bench_res_dir}/pow_and_calib.csv")
# subset = big_df[big_df['null']==False]
# plot_power(subset,bench_res_dir,'pow_plot')
















