import pandas as pd
from kgformula.post_process_plots import *
from kgformula.utils import x_q_class
import os
from generate_job_params import *
import ast
def concat_data(PATH,prefix,suffices):
    collection = []
    for el in suffices:
        collection.append(torch.load(PATH+prefix+el+'.pt'))
    return torch.cat(collection,dim=0).numpy()

def get_w_plot(data_path,est,w_est_path,args,pre_path,suffix):
    X, Y, Z, w = torch.load(data_path + '/data_seed=0.pt')
    if est:
        w_est = torch.load(w_est_path, map_location=torch.device('cpu'))
    else:
        Xq_class = x_q_class(qdist= args['qdist'], q_fac=args['q_factor'], X=X)
        X_q = Xq_class.sample(n=X.shape[0])
        w_est = Xq_class.calc_w_q(w)
    plt.hist(w_est.cpu().numpy(), 25)
    plt.hist(w.cpu().numpy(), 25,alpha=0.3)
    plt.savefig(pre_path+f'W_{suffix}.jpg')
    plt.clf()
    eff_est = calc_eff(w_est)
    return eff_est

def get_hist(ref_vals,name,pre_path,suffix):
    plt.hist(ref_vals, 25)
    plt.savefig(pre_path+f'{name}_{suffix}.jpg')
    plt.clf()

def return_filenames(args):
    estimate =  args['estimate']
    job_dir =  args['job_dir']
    data_dir =  args['data_dir']
    seeds_a =  args['seeds_a']
    seeds_b =  args['seeds_b']
    q_fac =  args['q_factor']
    qdist =  args['qdist']
    bootstrap_runs =  args['bootstrap_runs']
    est_params =  args['est_params']
    estimator =  args['estimator']
    runs =  args['runs']
    mode =  args['mode']
    split_data =  args['split']
    required_n =  args['n']
    suffix = f'_qf={q_fac}_qd={qdist}_m={mode}_s={seeds_a}_{seeds_b}_e={estimate}_est={estimator}_sp={split_data}_br={bootstrap_runs}_n={required_n}'

    p_val_file = f'./{data_dir}/{job_dir}/p_val_array{suffix}.pt'
    ref_val = f'./{data_dir}/{job_dir}/ref_val_array{suffix}.pt'
    w_file = f'./{data_dir}/{job_dir}/w_estimated{suffix}.pt'

    return p_val_file,ref_val,w_file,data_dir,job_dir,suffix,estimate

def data_dir_extract(data_dir):
    str_list = data_dir.split('/')[1].split('_')
    str_list = [str_list[i].split('=')[1] for i in [1 , 3,5,7,11]]
    str_list = [ast.literal_eval(x) for x in str_list]
    return str_list


def calc_eff(w):
    return (w.sum()**2)/(w**2).sum()

if __name__ == '__main__':
    jobs = os.listdir("job_dir")
    jobs.sort()
    df_data = []
    levels = [1e-3, 1e-2, 0.05, 1e-1]

    for j in jobs:
        job_params = load_obj(j, folder='job_dir/')
        p_val_file, ref_val, w_file, data_dir, job_dir, suffix, estimate =  return_filenames(job_params)
        pre_path = f'./{data_dir}/{job_dir}/'
        dat_param = data_dir_extract(data_dir)
        row = [dat_param[-1],job_params['q_factor'],job_params['qdist'],job_params['est_params']['n_sample'],job_params['estimator'],dat_param[3],dat_param[0][1]]

        try:
            eff_est = get_w_plot(data_path=data_dir,est=estimate,w_est_path=w_file,args=job_params,pre_path=pre_path,suffix=suffix)
            row.append(eff_est.item())
            pval_dist = torch.load(p_val_file).numpy()
            stat, pval = kstest(pval_dist, 'uniform')
            ks_test = pd.DataFrame([[stat,pval]],columns=['ks-stat','ks-pval'])
            get_hist(pval_dist,name='pvalhsit_',pre_path=pre_path,suffix=suffix)
            row.append(pval)
            custom_metric = pval_dist.mean()-0.5
            row.append(custom_metric)

            ref_vals = torch.load(ref_val).numpy()
            get_hist(ref_vals,name='rvalhsit_',pre_path=pre_path,suffix=suffix)
            h_0 = dat_param[0][1] == 0
            str_mode ='size' if h_0 else 'power'
            results_size = []
            for alpha in levels:
                power = get_power(alpha,pval_dist)
                results_size.append(power)
            row = row + results_size
            df_data.append(row)
            print('success')
        except Exception as e:
            print(e)
    columns  = ['$/beta_{xz}$','$c_q$','q_d','# perm','nce_style','d_Z','beta_xy' ,'EFF est w','KS pval','uniform-dev'] + [f'p_a={el}' for el in levels]
    df = pd.DataFrame(df_data,columns=columns)
    df = df.drop_duplicates()
    df = df.sort_values('KS pval',ascending=False)
    df.to_csv(f'post_process.csv')
    print(df.to_latex(escape=True))
