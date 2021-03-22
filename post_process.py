import pandas as pd
from kgformula.post_process_plots import *
from kgformula.utils import x_q_class
import os
from generate_job_params import *
import ast
from generate_data_multivariate import generate_sensible_variables,calc_snr
from pylab import *
rc('text', usetex=True)

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def concat_data(PATH,prefix,suffices):
    collection = []
    for el in suffices:
        collection.append(torch.load(PATH+prefix+el+'.pt'))
    return torch.cat(collection,dim=0).numpy()

def get_w_plot(data_path,est,w_est_path,args,pre_path,suffix):
    X, Y, Z, inv_w = torch.load(data_path + '/data_seed=0.pt')

    if est:
        w_est = torch.load(w_est_path, map_location=torch.device('cpu'))
    else:
        Xq_class =  x_q_class(qdist=args['qdist'], q_fac=args['q_factor'], X=X[:args['n'],:])
        w_est = Xq_class.calc_w_q(inv_w[:args['n']])

    X_class = x_q_class(qdist=args['qdist'], q_fac=args['q_factor'], X=X[:args['n'],:])
    w = X_class.calc_w_q(inv_w[:args['n']])
    try:
        plt.hist(w_est.cpu().numpy(), 25)
        plt.savefig(pre_path+f'W_{suffix}.jpg')
        plt.clf()
        plt.hist(w.cpu().numpy(), 25)
        plt.savefig(pre_path + f'realW_{suffix}.jpg')
        plt.clf()
    except Exception as e:
        print(e)

    try:
        if est:
            chunks = np.array_split(w.cpu().numpy(), 2)
            plt.scatter(chunks[-1],w_est.cpu().numpy())
            cor_coef = np.corrcoef(chunks[-1],w_est.cpu().numpy())
        else:
            plt.scatter(w.cpu().numpy(), w_est.cpu().numpy())
            cor_coef = np.corrcoef(w.cpu().numpy(), w_est.cpu().numpy())
        c = cor_coef[0][1]
        plt.suptitle(f"correlation: {c}")
        plt.savefig(pre_path + f'corr_{suffix}.jpg')
        plt.clf()
    except Exception as e:
        c=0


    eff_est = calc_ess(w_est)
    return eff_est,c

def get_hist(ref_vals,name,pre_path,suffix,args,snr,ess,bxy,ks_val):
    try:
        # n = args['n']
        # ess = round(ess,2)
        # snr = round(snr,3)
        # ks_val = round(ks_val,3)
        # if bxy==0:
        #     title = f'n: {n} ESS: {ess} SNR: {snr} KS-pval: {ks_val}'
        # else:
        #     title = f'n: {n} ESS: {ess} SNR: {snr}'
        #
        # if args['estimator']=='real_TRE_Q':
        #     estimator = 'TRE-Q'
        # elif  args['estimator']=='NCE_Q':
        #     estimator  ='NCE-Q'
        # else:
        #     estimator = args['estimator'].replace('_',' ')
        #
        # # xlabl =  r'Estimator: {est} $\quad \beta_{XY}={bxy}$'.format(est=estimator,bxy=bxy,XY='{XY}')
        plt.hist(ref_vals, 25)
        # plt.suptitle(title)
        plt.xlabel('p-values')
        plt.ylabel('Frequency')
        # plt.show()
        plt.savefig(pre_path+f'{name}_{suffix}.jpg')
        plt.clf()
    except Exception as e:
        print(e)

def return_filenames(args):
    estimate =  args['estimate']
    job_dir =  args['job_dir']
    data_dir =  args['data_dir']
    seeds_a =  args['seeds_a']
    seeds_b =  args['seeds_b']
    q_fac =  args['q_factor']
    qdist =  args['qdist']
    bootstrap_runs =  args['bootstrap_runs']
    estimator =  args['estimator']
    mode =  args['mode']
    split_data =  args['split']
    required_n =  args['n']
    suffix = f'_qf={q_fac}_qd={qdist}_m={mode}_s={seeds_a}_{seeds_b}_e={estimate}_est={estimator}_sp={split_data}_br={bootstrap_runs}_n={required_n}'

    p_val_file = f'./{data_dir}/{job_dir}/p_val_array{suffix}.pt'
    ref_val = f'./{data_dir}/{job_dir}/ref_val_array{suffix}.pt'
    w_file = f'./{data_dir}/{job_dir}/w_estimated{suffix}.pt'
    validity_pval = f'./{data_dir}/{job_dir}/validity_p_value_array{suffix}.pt'
    validity_vals = f'./{data_dir}/{job_dir}/validity_value_array{suffix}.pt'
    actual_validity_pvals = f'./{data_dir}/{job_dir}/actual_validity_p_value_array{suffix}.pt'

    return p_val_file,ref_val,w_file,data_dir,job_dir,suffix,estimate,validity_pval,validity_vals,actual_validity_pvals

def data_dir_extract(data_dir):
    str_list = data_dir.split('/')[1].split('_')
    str_list = [str_list[i].split('=')[1] for i in [1 , 3,5,7,11]]
    str_list = [ast.literal_eval(x) for x in str_list]
    return str_list


def calc_ess(w):
    return (w.sum()**2)/(w**2).sum()

def calculate_one_row(j,base_dir):
    levels = [1e-3, 1e-2,0.025, 0.05, 1e-1]
    theta_dict = {1: 2.0, 3: 3.0, 15: 8.0, 50: 16.0}
    job_params = load_obj(j, folder=f'{base_dir}/')
    try:
        p_val_file, ref_val, w_file, data_dir, job_dir, suffix, estimate,validity_pval_filename,validity_val_filename,actual_validity_fname = return_filenames(job_params)
        pre_path = f'./{data_dir}/{job_dir}/'
        dat_param = data_dir_extract(data_dir)
        bxz = dat_param[-1]
        d_Z = dat_param[3]
        b_z = (d_Z ** 2) * bxz
        b_z = generate_sensible_variables(d_Z, b_z, 0)
        snr_xz = calc_snr(b_z, theta_dict[d_Z])
        bxy = dat_param[0][1]
        row = [job_params['n'], bxz, job_params['q_factor'], job_params['qdist'], job_params['est_params']['n_sample'],
               job_params['estimator'], d_Z, bxy, snr_xz]

        eff_est, corr_coeff = get_w_plot(data_path=data_dir, est=estimate, w_est_path=w_file, args=job_params,
                                         pre_path=pre_path, suffix=suffix)
        row.append(eff_est.item())
        row.append(corr_coeff)
        pval_dist = torch.load(p_val_file).numpy()
        stat, pval = kstest(pval_dist, 'uniform')
        get_hist(pval_dist, name='pvalhsit_', pre_path=pre_path, suffix=suffix, args=job_params, snr=snr_xz,
                 ess=eff_est.item(), bxy=bxy, ks_val=pval)
        row.append(pval)
        row.append(stat)
        custom_metric = pval_dist.mean() - 0.5
        row.append(custom_metric)

        ref_vals = torch.load(ref_val).numpy()
        get_hist(ref_vals, name='rvalhsit_', pre_path=pre_path, suffix=suffix, args=job_params, snr=snr_xz,
                 ess=eff_est.item(), bxy=bxy, ks_val=pval)
        h_0 = dat_param[0][1] == 0
        results_size = []
        for alpha in levels:
            power = get_power(alpha, pval_dist)
            results_size.append(power)
        row = row + results_size
        try:
            validity_pval = torch.load(validity_pval_filename).numpy()
            get_hist(validity_pval, name='validity_pvals_', pre_path=pre_path, suffix=suffix, args=job_params, snr=snr_xz,
                     ess=eff_est.item(), bxy=bxy, ks_val=pval)
            validity_val = torch.load(validity_val_filename).numpy()
            get_hist(validity_val, name='validity_val_', pre_path=pre_path, suffix=suffix, args=job_params, snr=snr_xz,
                 ess=eff_est.item(), bxy=bxy, ks_val=pval)
            actual_validity_pvals = torch.load(actual_validity_fname).numpy().squeeze()
            get_hist(actual_validity_pvals, name='actual_validity_pval_', pre_path=pre_path, suffix=suffix, args=job_params, snr=snr_xz,
                 ess=eff_est.item(), bxy=bxy, ks_val=pval)

        except Exception as e:
            print(e)
            print('dont have validity step')

        print('success')
        return row
    except Exception as e:
        print(e)

def generate_csv_file(base_dir):
    jobs = os.listdir(base_dir)
    jobs.sort()
    df_data = []
    for j in jobs:
        row = calculate_one_row(j,base_dir)
        if isinstance(row,list):
            df_data.append(row)
    levels = [1e-3, 1e-2,0.025, 0.05, 1e-1]
    columns  = ['n','$/beta_{xz}$','$c_q$','q_d','# perm','nce_style','d_Z','beta_xy','snr_xz','EFF est w','true_w_q_corr','KS pval','KS stat','uniform-dev'] + [f'p_a={el}' for el in levels]
    df = pd.DataFrame(df_data,columns=columns)
    df = df.drop_duplicates()
    df = df.sort_values('KS pval',ascending=False)
    df.to_csv(f'{base_dir}.csv')
    print(df.to_latex(escape=True))

def multi_run_wrapper(args):
   return calculate_one_row(*args)

def generate_csv_file_parfor(base_dir):
    import multiprocessing as mp
    pool = mp.Pool(mp.cpu_count())
    jobs = os.listdir(base_dir)
    jobs.sort()
    df_data = pool.map(multi_run_wrapper, [(row,base_dir) for row in jobs])
    df_data = list(filter(None, df_data))
    # df_data = [pool.apply(calculate_one_row, args=(row, base_dir)) for row in jobs]
    pool.close()
    levels = [1e-3, 1e-2,0.025, 0.05, 1e-1]
    columns  = ['n','$/beta_{xz}$','$c_q$','q_d','# perm','nce_style','d_Z','beta_xy','snr_xz','EFF est w','true_w_q_corr','KS pval','KS stat','uniform-dev'] + [f'p_a={el}' for el in levels]
    df = pd.DataFrame(df_data,columns=columns)
    df = df.drop_duplicates()
    df = df.sort_values('KS pval',ascending=False)
    df.to_csv(f'{base_dir}.csv')

if __name__ == '__main__':
    # generate_csv_file_parfor('base_jobs_kc_est')
    generate_csv_file_parfor('base_jobs_kc')
    generate_csv_file_parfor('base_jobs_kc_est_rulsif')
