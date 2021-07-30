from scipy.stats import kstest
from create_plots import *


def plot_power(raw_df,dir,name):
    for d in [1]:
        for alp in [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]:
            subset_3 = raw_df[raw_df['alp']==alp]
            a,b,e = calc_error_bars(subset_3['alp=0.05'],alpha=0.05,num_samples=100)
            plt.plot('n','alp=0.05',data=subset_3,linestyle='--', marker='o',label=r'$\gamma'+f'={alp}$')
            plt.fill_between(subset_3['n'], a, b, alpha=0.1)
    plt.hlines(0.05, 0, 10000)
    plt.legend(prop={'size': 10})
    plt.xticks([1000,5000,10000])
    plt.xlabel(r'$n$')
    plt.ylabel(r'Power $\alpha=0.05$')
    plt.savefig(f'{dir}/{name}.png',bbox_inches = 'tight',pad_inches = 0.05)
    plt.clf()


def calc_power(vec, level=.05):
    n = vec.shape[0]
    pow = np.sum(vec<=level)/n
    return pow


bench_res_dir = '1d_cat_pow'
benchmark_data = pd.read_csv('hdm_bench_syntehtic.csv')
if not os.path.exists(bench_res_dir):
    os.makedirs(bench_res_dir)
# print(benchmark_data)
bench_extract_cols = list(str(el) for el in range(1,101))
# bench_submat_false = benchmark_data[benchmark_data['null']==False]
# for n in [1000,5000,10000]:
#     for alp in [0.00,0.02,0.04,0.06,0.08,0.10]:

pow_and_calib = []
for i,row in benchmark_data.iterrows():
    data_row = []
    p_vals = row[bench_extract_cols].values.astype(float).squeeze()
    if row['null']==False:
        for lvl in [0.01,0.05,0.1]:
            pow = calc_power(p_vals,lvl)
            data_row.append(pow)
    else:
        _,p_val_ks_test = kstest(p_vals,'uniform')
        data_row = [p_val_ks_test]*3
    pow_and_calib.append(data_row)

df_calib_pow = pd.DataFrame(pow_and_calib,columns=['alp=0.01','alp=0.05','alp=0.1'])
big_df = pd.concat([benchmark_data,df_calib_pow],axis=1)
big_df.to_csv(f"{bench_res_dir}/pow_and_calib.csv")
subset = big_df[big_df['null']==False]
plot_power(subset,bench_res_dir,'pow_plot')
subset = big_df[big_df['null']==True]
plot_power(subset,bench_res_dir,'calib_plot')















