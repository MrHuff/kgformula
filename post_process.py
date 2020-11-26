import pandas as pd
from kgformula.post_process_plots import *
from kgformula.utils import x_q_class

mode='Q'

prefix_pval = 'p_val_array_'
prefix_ref = 'ref_val_array_'
prefix_hsic_pval = 'hsic_pval_array_'

h_0=True
p_val = True
size = True
ref_val = False
hsic_val = False
seed_max = 100


def concat_data(PATH,prefix,suffices):
    collection = []
    for el in suffices:
        collection.append(torch.load(PATH+prefix+el+'.pt'))
    return torch.cat(collection,dim=0).numpy()
def return_file_name(q_fac,q_dist,a,b,perm,br,variant,nce,est):

    _tmp = f'qf={q_fac}_qd={q_dist}_m={mode}_s={a}_{b}_e={est}_est={nce}_sp={est}_p={perm}_br={br}_v={variant}'

    # f'm=Q_s={a}_{b}_e=False_est=NCE_sp=False_p={perm}'
    #f'm=new_s={a}_{b}_e=True_est=NCE_sp=True_lr=0.0001_ma=5000_wi=32_la=4_mi=False_bs=0.001_va=0.01_n_=250_cr=0.05_ki=10_ka=10_m=10000_d_=3_d_=3'
    # f'm=Q_s={a}_{b}_e=True_est=NCE_sp=True_lr=0.0001_ma=5000_wi=32_la=4_mi=False_bs=0.001_va=0.01_n_=250_cr=0.05_ki=10_ka=10_m=10000_d_=3_d_=3'
    #f'm=regular_s={a}_{b}_e=False_est=NCE_sp=False'
    #f'm=new_s={a}_{b}_e=True_est=NCE_sp=True_lr=0.0001_ma=5000_wi=32_la=4_mi=False_bs=0.001_va=0.01_n_=250_cr=0.05_ki=10_ka=10_m=10000_d_=1_d_=1'
    return _tmp

def calc_eff(w):
    return (w.sum()**2)/(w**2).sum()

if __name__ == '__main__':
    n=10000
    levels = [1e-3, 1e-2, 0.05, 1e-1]
    for est in [True]:
        df_data = []
        for q in [1.0]:
            for q_d in [2]:
                for bootstrap_runs in [250]:
                    for variant in [1]:
                        for d_X, d_Y, d_Z, theta, phi in zip([1, 3], [1, 3], [1, 3], [2.0, 2.0], [2.0, 2.0]):
                            for beta_xz in [0.25, 0.5, 0.01, 0.1, 0.0]:
                                for perm in ['Y']:
                                    for nce in ['NCE_Q']:
                                        for by in [1e-3,1e-2,1e-1,0.1,0.25,0.5]:
                                            row = [perm,beta_xz,q,q_d,bootstrap_runs,variant,nce,d_Z,by]
                                            h_int = int(not by==1)
                                            ba = 0.0
                                            if d_X == 1 and by == 0.0:
                                                by = 0
                                            if d_X == 3:
                                                ba = 0
                                            beta_xy = [ba, by]
                                            data_path = f"data_100/beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n={n}_yz=0.5_beta_XZ={beta_xz}_theta={theta}_phi={phi}/"
                                            PATH = data_path + 'layers=3_width=32/'
                                            f_name=return_file_name(q,q_d,0,100,perm,bootstrap_runs,variant,nce,est)
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
                                                        suffices.append(return_file_name(q,q_d,a,b,perm,bootstrap_runs,variant,nce,est))
                                                    if est:
                                                        str_= suffices[0]
                                                        load_name = f'w_estimated_{str_}.pt'
                                                        w_est = torch.load(PATH+load_name,map_location=torch.device('cpu'))

                                                    else:
                                                        X,Y,Z,w=torch.load(data_path+'data_seed=0.pt')
                                                        Xq_class = x_q_class(qdist=q_d, q_fac=q, X=X)
                                                        X_q = Xq_class.sample(n=X.shape[0])
                                                        w_est = Xq_class.calc_w_q(w)
                                                    plt.hist(w_est.cpu().numpy(), 25)
                                                    plt.savefig(PATH + f'{f_name}_weights.png')
                                                    plt.clf()
                                                    eff_est = calc_eff(w_est)
                                                    row.append(eff_est.item())
                                                    if p_val:
                                                        pval_dist = concat_data(PATH,prefix_pval,suffices)
                                                        stat, pval = kstest(pval_dist, 'uniform')
                                                        ks_test = pd.DataFrame([[stat,pval]],columns=['ks-stat','ks-pval'])
                                                        ks_test.to_csv(PATH+'fdf_'+f_name+'.csv')
                                                        plt.hist(pval_dist,25)
                                                        plt.savefig(PATH+f'pval_{q}_{beta_xz}'+f_name+'.jpg')
                                                        plt.clf()
                                                        row.append(pval)
                                                        custom_metric = pval_dist.mean()-0.5
                                                        row.append(custom_metric)
                                                    if ref_val:
                                                        ref_vals = concat_data(PATH,prefix_pval,suffices)
                                                        plt.hist(ref_vals,25)
                                                        plt.savefig(PATH+f'refval_{q}_{beta_xz}'+f_name+'.jpg')
                                                        plt.clf()

                                                    if size:
                                                        pval_dist = concat_data(PATH,prefix_pval,suffices)
                                                        str_mode ='size' if h_0 else 'power'
                                                        results_size = []
                                                        levels = [1e-3, 1e-2, 0.05, 1e-1]
                                                        for alpha in levels:
                                                            power = get_power(alpha,pval_dist)
                                                            results_size.append(power)
                                                        row = row + results_size
                                                    print('success')
                                                    df_data.append(row)
                                                    break
                                                except Exception as e:
                                                    print(e)
        columns  = ['perm','$/beta_{xz}$','$c_q$','q_d','# perm','vari','nce_style','d_Z','beta_xy' ,'EFF est w','KS pval','uniform-dev'] + [f'p_a={el}' for el in levels ]
        df = pd.DataFrame(df_data,columns=columns)
        df = df.drop_duplicates()
        df = df.sort_values('KS pval',ascending=False)
        df.to_csv(f'post_process.csv')
        print(df.to_latex(escape=True))
