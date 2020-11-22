import pandas as pd
from kgformula.post_process_plots import *


mode='Q'

prefix_pval = 'p_val_array_'
prefix_ref = 'ref_val_array_'
prefix_hsic_pval = 'hsic_pval_array_'

h_0=True
p_val = True
size = True
histogram_true = True
ref_val = False
hsic_val = False
seed_max = 100


def concat_data(PATH,prefix,suffices):
    collection = []
    for el in suffices:
        collection.append(torch.load(PATH+prefix+el+'.pt'))
    return torch.cat(collection,dim=0).numpy()
def return_file_name(a,b,perm,br,variant,nce,est):

    _tmp = f'm={mode}_s={a}_{b}_e={est}_est={nce}_sp={est}_p={perm}_br={br}_v={variant}'
        # f'm=Q_s={a}_{b}_e=False_est=NCE_sp=False_p={perm}'
    #f'm=new_s={a}_{b}_e=True_est=NCE_sp=True_lr=0.0001_ma=5000_wi=32_la=4_mi=False_bs=0.001_va=0.01_n_=250_cr=0.05_ki=10_ka=10_m=10000_d_=3_d_=3'
    # f'm=Q_s={a}_{b}_e=True_est=NCE_sp=True_lr=0.0001_ma=5000_wi=32_la=4_mi=False_bs=0.001_va=0.01_n_=250_cr=0.05_ki=10_ka=10_m=10000_d_=3_d_=3'
    #f'm=regular_s={a}_{b}_e=False_est=NCE_sp=False'
    #f'm=new_s={a}_{b}_e=True_est=NCE_sp=True_lr=0.0001_ma=5000_wi=32_la=4_mi=False_bs=0.001_va=0.01_n_=250_cr=0.05_ki=10_ka=10_m=10000_d_=1_d_=1'
    return _tmp

def calc_eff(w):
    return (w.sum()**2)/(w**2).sum()

if __name__ == '__main__':
    for est in [True]:
        df_data = []
        for q in [1.0]:
            for bootstrap_runs in [250]:
                for variant in [1]:
                    for d_Z, theta, phi in zip([1, 3, 50], [2.0, 2.0, 8.0],[2.0, 2.0, 2.0]):  # zip([1],[2.0],[2.0]):
                        for beta_xz in [0.25, 0.5, 0.01, 0.1, 0.0]:
                            for perm in ['Y']:
                                for nce in ['NCE']:
                                    for beta_xy in [1e-3,1e-2,1e-1,0.1,0.25,0.5]:
                                        row = [perm,beta_xz,q,bootstrap_runs,variant,nce,d_Z,beta_xy]
                                        h_int = int(not beta_xy==1)
                                        mv_str = f'q={q}_mv_100/beta_xy=[0, {beta_xy}]_d_X=3_d_Y=3_d_Z={d_Z}_n=10000_yz=0.5_beta_XZ={beta_xz}_theta={theta}_phi={phi}/'
                                        uni_str =  f'univariate_100_seeds/Q={q}_gt=H_{h_int}_y_a=0.0_y_b={beta_xy}_z_a=0.0_z_b={beta_xz}_cor=0.5_n=10000_seeds=100_{theta}_{phi}/'
                                        if d_Z==1:
                                            data_path = uni_str
                                        else:
                                            data_path = mv_str
                                            nr_of_gpus=3

                                        PATH = data_path + 'layers=2_width=32/'
                                        f_name=return_file_name(0,100,perm,bootstrap_runs,variant,nce,est)
                                        if histogram_true:
                                            X, Y, Z, X_q, _w, w_q = torch.load(data_path+'data_seed=0.pt')
                                            w_eff = round(calc_eff(_w).item(),2)
                                            w_q_eff = round(calc_eff(w_q).item(),2)
                                            row.append(w_q_eff)
                                            plt.hist(_w,250)
                                            plt.suptitle(f'true weights w_i, EFF={w_eff}')
                                            plt.savefig(PATH+f'w_i_{q}_{beta_xz}'+f_name+'.jpg')
                                            plt.clf()
                                            plt.hist(w_q,250)
                                            plt.suptitle(f'true weights w_i^q, EFF={w_q_eff}')
                                            plt.savefig(PATH+f'w_i^q_{q}_{beta_xz}'+f_name+'.jpg')
                                            plt.clf()

                                        for nr_of_gpus in [1,2,3,4,5,6,7,8]:
                                            try:
                                                residual = seed_max % nr_of_gpus
                                                interval_size = (seed_max - residual) / nr_of_gpus
                                                suffices = []
                                                    #f'univariate_100_seeds/Q={q}_gt=H_0_y_a=0.0_y_b=0.0_z_a=0.0_z_b={beta_xz}_cor=0.5_n=10000_seeds=100_4_2.0/'
                                                for i in range(nr_of_gpus):
                                                    if i == nr_of_gpus - 1:
                                                        a = int(i * interval_size)
                                                        b = int((i + 1) * interval_size + residual)
                                                    else:
                                                        a = int(i * interval_size)
                                                        b = int((i + 1) * interval_size )
                                                    suffices.append(return_file_name(a,b,perm,bootstrap_runs,variant,nce,est))

                                                if p_val:
                                                    pval_dist = concat_data(PATH,prefix_pval,suffices)
                                                    stat, pval = kstest(pval_dist, 'uniform')
                                                    ks_test = pd.DataFrame([[stat,pval]],columns=['ks-stat','ks-pval'])
                                                    ks_test.to_csv(PATH+'fdf_'+f_name+'.csv')
                                                    plt.hist(pval_dist,25)
                                                    plt.savefig(PATH+f'pval_{q}_{beta_xz}'+f_name+'.jpg')
                                                    plt.clf()
                                                    row.append(pval)

                                                if ref_val:
                                                    ref_vals = concat_data(PATH,prefix_pval,suffices)
                                                    plt.hist(ref_vals,25)
                                                    plt.savefig(PATH+f'refval_{q}_{beta_xz}'+f_name+'.jpg')
                                                    plt.clf()

                                                if size:
                                                    pval_dist = concat_data(PATH,prefix_pval,suffices)
                                                    levels = [1e-3,1e-2,0.05,1e-1]
                                                    str_mode ='size' if h_0 else 'power'
                                                    results_size = []
                                                    for alpha in levels:
                                                        power = get_power(alpha,pval_dist)
                                                        results_size.append(power)
                                                    row = row + results_size
                                                df_data.append(row)
                                                print('success')
                                                break
                                            except Exception as e:
                                                print(e)
                                        df_data.append(row)
        columns  = ['perm','$/beta_{xz}$','$c_q$','# perm','vari','nce_style','d_Z','beta_xy' ,'EFF w_q','KS pval'] + [f'power_alpha={el}' for el in levels ]
        df = pd.DataFrame(df_data,columns=columns)
        df = df.drop_duplicates()
        df = df.sort_values('KS pval',ascending=False)
        df.to_csv(f'post_process.csv')
        print(df.to_latex(escape=True))
