import argparse
from matplotlib import pyplot as plt
from scipy.stats import kstest
import tqdm
import pandas as pd
import torch
from kgformula.test_statistics import *
from kgformula.fixed_do_samplers import simulate_xyz_univariate,apply_qdist
import os
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.tri as mtri
import time
from kgformula.test_statistics import  hsic_sanity_check_w,hsic_test

def EFF_calc(w):
    return w.sum()**2/(w**2).sum()

def true_dens_plot(var,M,dist_a,dist_b):
    x, z = np.meshgrid(np.linspace(-var * 5, var * 5, 100), np.linspace(-var * 5, var * 5, 100))
    val = torch.stack([torch.from_numpy(x), torch.from_numpy(z)], dim=-1).float()
    w_true_plt = (-M.log_prob(val) + (dist_a.log_prob(torch.from_numpy(x).float()) + dist_b.log_prob(
        torch.from_numpy(z).float()))).exp().numpy()
    fig, ax = plt.subplots()
    c = ax.pcolormesh(x, z, w_true_plt, cmap='RdBu', vmin=w_true_plt.min(), vmax=w_true_plt.max())
    fig.colorbar(c, ax=ax)
    plt.show()

def experiment_plt_do(w_true,w_classify,X,Z,title):

    def tricol_plt(ax,name,triang,w):
        p = ax.tricontourf(triang,w)
        plt.colorbar(p,ax=ax)
        ax.title.set_text(name)

    def hist(ax,name,w):
        ax.hist(w)
        ax.title.set_text(name)

    X = X.cpu().flatten().numpy()
    Z = Z.cpu().flatten().numpy()

    w_true = w_true.cpu().flatten().numpy()
    w_classify = w_classify.cpu().flatten().numpy()
    fig, axs = plt.subplots(1, 4, figsize=(40, 8))
    fig.tight_layout()
    fig.suptitle(title)
    triang = mtri.Triangulation(X, Z)

    tricol_plt(axs[0],'w_true',triang,w_true)
    tricol_plt(axs[1],'w_estimate',triang,w_classify)
    hist(axs[2],'w_true_hist',w_true)
    hist(axs[3],'w_estimate_hist',w_classify)
    plt.subplots_adjust(top=0.85)
    plt.savefig(title+'.png')


def experiment_plt_mv(w_true,w_classify,X,Z,title,var,M,dist_a,dist_b,model):

    def hist(ax,name,w):
        ax.hist(w,bins=50)
        ax.title.set_text(name)

    w_true = w_true.cpu().flatten().numpy()
    w_classify = w_classify.cpu().flatten().numpy()
    fig, axs = plt.subplots(1, 2, figsize=(40, 8))
    fig.tight_layout()
    fig.suptitle(title)
    hist(axs[0],'w_true_hist',w_true)
    hist(axs[1],'w_estimate_hist',w_classify)
    plt.subplots_adjust(top=0.85)
    plt.savefig(title+'.png')

def experiment_plt(w_true,w_classify,X,Z,title,var,M,dist_a,dist_b,model):

    def get_data(var):
        x, z = np.meshgrid(np.linspace(-var * 4, var * 4, 100), np.linspace(-var * 4, var * 4, 100))
        val = torch.stack([torch.from_numpy(x), torch.from_numpy(z)], dim=-1).float()
        return x,z,val

    def true_w(x,z,val,M,dist_a,dist_b):
        return (-M.log_prob(val) + (dist_a.log_prob(torch.from_numpy(x).float()) + dist_b.log_prob(
        torch.from_numpy(z).float()))).exp().numpy()

    def pred_w(x,z,model):
        p_w = model.get_w(torch.from_numpy(x).float().view(-1,1).cuda(),torch.from_numpy(z).flatten().float().view(-1,1).cuda()).cpu().numpy()
        return p_w.reshape(100,100)

    def tricol_plt(ax,name,triang,w):
        p = ax.tricontourf(triang,w)
        plt.colorbar(p,ax=ax)
        ax.title.set_text(name)

    def hist(ax,name,w):
        ax.hist(w)
        ax.title.set_text(name)

    def plt_dense_w(ax,name,x,z,w):
        c = ax.pcolormesh(x, z, w, cmap='RdBu', vmin=w.min(), vmax=w.max())
        plt.colorbar(c, ax=ax)
        ax.title.set_text(name)

    X = X.cpu().flatten().numpy()
    Z = Z.cpu().flatten().numpy()
    x,z,val = get_data(var)
    dens_w = true_w(x,z,val,M,dist_a,dist_b)
    with torch.no_grad():
        dens_w_pred = pred_w(x,z,model).squeeze()

    w_true = w_true.cpu().flatten().numpy()
    w_classify = w_classify.cpu().flatten().numpy()
    fig, axs = plt.subplots(1, 6, figsize=(40, 8))
    fig.tight_layout()
    fig.suptitle(title)
    triang = mtri.Triangulation(X, Z)

    tricol_plt(axs[0],'w_true',triang,w_true)
    tricol_plt(axs[1],'w_estimate',triang,w_classify)
    hist(axs[2],'w_true_hist',w_true)
    hist(axs[3],'w_estimate_hist',w_classify)
    plt_dense_w(axs[4],'true_dense_w_true',x,z,dens_w)
    plt_dense_w(axs[5],'pred_dense_w_est',x,z,dens_w_pred)
    plt.subplots_adjust(top=0.85)
    plt.savefig(title+'.png')

def debug_W(w,str):
    plt.hist(w.flatten().cpu().numpy(), 100)
    plt.title(str)
    plt.show()
    plt.clf()
    print(f'{str} EFF',EFF_calc(w))
    print(f'{str} max_val: ',w.max())
    print(f'{str} min_val: ',w.min())
    print(f'{str} var: ',w.var())
    print(f'{str} median:  ',w.median())

def get_density_plot(w,X,Z,ax,title=''):
    X = X.cpu().numpy().flatten()
    Z = Z.cpu().numpy().flatten()
    triang = mtri.Triangulation(X, Z)
    plt.tricontourf(triang, w)
    plt.colorbar()
    plt.title(title)
    plt.show()
    plt.clf()

def get_w_estimate_and_plot(X,Z,est_params,estimator,device):
    d = density_estimator(x=X, z=Z, cuda=True, est_params=est_params, type=estimator,  device=device)
    return d

def load_csv(path, d_Z,device):
    ls_Z = [f'z{i}' for i in range(1,d_Z+1)]
    dat = pd.read_csv(path) #Seems like I've screwed up!
    X = torch.from_numpy(dat['x'].values).float().unsqueeze(-1).cuda(device)
    Y = torch.from_numpy(dat['y'].values).float().unsqueeze(-1).cuda(device)
    Z = torch.from_numpy(dat[ls_Z].values).float().cuda(device)
    w = torch.from_numpy(dat['w'].values).float().cuda(device)
    return X,Y,Z,1/w

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def ecdf(data):
    """ Compute ECDF """
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return(x,y)

def generate_data(y_a,y_b,z_a,z_b,cor,n,seeds,theta=4,phi=2.0,q_factor=0.5):
    beta = {'y':[y_a,y_b],'z':[z_a,z_b]}
    if y_b == 0:
        ground_truth = 'H_0'
    else:
        ground_truth = 'H_1'
    data_dir = f'univariate_{seeds}_seeds/Q={q_factor}_gt={ground_truth}_y_a={y_a}_y_b={y_b}_z_a={z_a}_z_b={z_b}_cor={cor}_n={n}_seeds={seeds}_{theta}_{round(phi,2)}'
    if not os.path.exists(f'./{data_dir}/'):
        os.makedirs(f'./{data_dir}/')
    for i in range(seeds):
        X, Y, Z, w,pden = simulate_xyz_univariate(n=n, beta=beta, cor=cor, fam=1, oversamp=10, seed=i,theta=theta,phi=phi,q_factor=q_factor)
        X_q,w_q = apply_qdist(inv_wts=w,q_factor=q_factor,theta=theta,X=X)
        with torch.no_grad():
            if i==0:
                sig_xxz = theta
                e_xz = torch.cat([torch.ones_like(Z), Z],dim=1) @ torch.tensor(beta['z']) #XZ dependence
                sample_X = (X-e_xz.unsqueeze(-1)).squeeze().numpy()#*sig_xxz
                stat,pval=kstest(sample_X,'norm',(0,sig_xxz))
                print(sig_xxz)
                print(f'KS-stat: {stat}, pval: {pval}')
                p_val = hsic_test(X, Z, 100)
                sanity_pval = hsic_sanity_check_w(w, X, Z, 100)
                print(f'HSIC X Z: {p_val}')
                print(f'sanity_check_w : {sanity_pval}')
                plt.hist(w_q.numpy(),bins=250)
                plt.savefig(f'./{data_dir}/w_q.png')
                plt.clf()
                plt.hist(w.numpy(),bins=250)
                plt.savefig(f'./{data_dir}/w.png')
                plt.clf()

        torch.save((X,Y,Z,X_q,w,w_q),f'./{data_dir}/data_seed={i}.pt')

def torch_to_csv(path,filename):
    X,Y,Z,X_q,w,w_q,pden,qden= torch.load(path+filename)

    df = pd.DataFrame(torch.cat([X,Y,Z,X_q,w.unsqueeze(-1),w_q.unsqueeze(-1),pden.unsqueeze(-1),qden.unsqueeze(-1)],dim=1).numpy(),columns=['X','Y','Z','X_q','w','w_q','pden','qden'])
    f = filename.split('.')[0]
    df.to_csv(path+f+'.csv')

def job_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, nargs='?')
    parser.add_argument('--estimate', default=False, help='estimate w',type=str2bool, nargs='?')
    parser.add_argument('--debug_plot', default=False, help='estimate w',type=str2bool, nargs='?')
    parser.add_argument('--cuda', default=True, help='cuda',type=str2bool, nargs='?')
    parser.add_argument('--seeds', type=int, nargs='?', default=1000, help='seeds')
    parser.add_argument('--bootstrap_runs', type=int, nargs='?', default=250, help='bootstrap_runs')
    parser.add_argument('--alpha', type=float, nargs='?', default=0.5, help='alpha')
    parser.add_argument('--estimator', type=str, nargs='?',default='kmm')
    parser.add_argument('--lamb', type=float, nargs='?', default=0.5, help='lamb')
    parser.add_argument('--runs', type=int, nargs='?', default=1, help='runs')

    return parser

def hypothesis_acceptance(power,alpha=0.05,null_hypothesis=True):

    if null_hypothesis: #If hypothesis is true
        if power > alpha:
            return 1 #Then we did the right thing!
        else:
            return 0
    else: #If the hypothesis is False
        if power < alpha: # And we observe a pretty extreme value
            return 1 #That's great!
        else:
            return 0

def reject_outliers(data, m = 2.):
    data = np.array(data)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def calculate_pval(bootstrapped_list, test_statistic):
    pval = 1-1/(bootstrapped_list.shape[0]+1) *(1 + (bootstrapped_list<=test_statistic).sum())
    return pval.item()

def get_median_and_std(data):
    with torch.no_grad():
        mean = data.mean(dim=0)
        median,_ = data.median(dim=0)
        std = data.var(dim=0)**0.5
        x = [i for i in range(1,data.shape[1]+1)]
        return x, mean.cpu().numpy(),std.cpu().numpy(),median.cpu().numpy()

def run_job_func(args):
    j = simulation_object(args)
    j.run()

def split(x,n_half):
    if len(x.shape)==1:
        return x[:n_half],x[n_half:]
    else:
        return x[:n_half,:],x[n_half:,:]

class simulation_object():
    def __init__(self,args):
        self.args=args
        self.cuda = self.args['cuda']
        self.device = self.args['device']

    def run(self):
        estimate = self.args['estimate']
        job_dir = self.args['job_dir']
        data_dir = self.args['data_dir']
        seeds_a = self.args['seeds_a']
        seeds_b = self.args['seeds_b']
        perm = self.args['perm']
        bootstrap_runs  = self.args['bootstrap_runs']
        est_params = self.args['est_params']
        estimator = self.args['estimator']
        runs = self.args['runs']
        mode = self.args['mode']
        split_data = self.args['split']
        variant = self.args['variant']
        ks_data = []
        R2_errors = []
        hsic_pval_list = []
        suffix = f'_m={mode}_s={seeds_a}_{seeds_b}_e={estimate}_est={estimator}_sp={split_data}_p={perm}_br={bootstrap_runs}_v={variant}'
        # if estimate:
        #     if estimator in ['NCE', 'TRE', 'linear_classifier']:
        #         for key,val in est_params.items():
        #             suffix = suffix + f'_{key[0:2]}={val}'

        if not os.path.exists(f'./{data_dir}/{job_dir}'):
            os.makedirs(f'./{data_dir}/{job_dir}')
        mse_loss = torch.nn.MSELoss()
        for j in range(runs):
            p_value_list = []
            reference_metric_list = []
            for i in tqdm.trange(seeds_a,seeds_b):
                if self.cuda:
                    X, Y, Z,X_q,_w,w_q  = torch.load(f'./{data_dir}/data_seed={i}.pt',map_location=f'cuda:{self.device}')
                else:
                    X, Y, Z,X_q,_w,w_q  = torch.load(f'./{data_dir}/data_seed={i}.pt')

                if split_data:
                    n_half = X.shape[0]//2
                    X_train,X_test = split(X,n_half)
                    Y_train,Y_test = split(Y,n_half)
                    Z_train,Z_test = split(Z,n_half)
                    X_q_train,X_q_test = split(X_q,n_half)
                    _,_w = split(_w,n_half)
                    _,w_q = split(w_q,n_half)
                else:
                    X_train= X
                    Z_train = Z
                    X_test = X
                    Z_test = Z
                    Y_test = Y
                    X_q_test = X_q

                if estimate:
                    d = density_estimator(x=X_train, z=Z_train, cuda=self.cuda,
                                          est_params=est_params, type=estimator, device=self.device)
                    w = d.return_weights(X_test,Z_test)
                    if estimator in ['NCE', 'TRE_Q','NCE_Q', 'linear_classifier']:
                        with torch.no_grad():
                            l = mse_loss(_w, w) / _w.var()
                        R2_errors.append(1-l.item())
                        hsic_pval_list.append(d.hsic_pval)
                    X_q_test = d.X_q_test
                    if j == 0:
                        torch.save(w,f'./{data_dir}/{job_dir}/w_estimated{suffix}.pt')
                else:
                    if mode=='Q':
                        w = w_q
                    elif mode=='new':
                        w = _w
                    elif mode=='regular':
                        w = _w
                if mode=='Q':
                    c = Q_weighted_HSIC(X=X_test, Y=Y_test, X_q=X_q_test, w=w, cuda=self.cuda, device=self.device,perm=perm,seed=i)
                elif mode=='new' :
                    c = weighted_HSIC(X=X_test, Y=Y_test, w=w, device=self.device,perm=perm,variant=variant)
                reference_metric = c.calculate_weighted_statistic().cpu().item()
                list_of_metrics = []
                for i in range(bootstrap_runs):
                    list_of_metrics.append(c.permutation_calculate_weighted_statistic().cpu().item())
                array = torch.tensor(list_of_metrics).float() #seem to be extremely sensitive to lengthscale, i.e. could be sign flipper
                p = calculate_pval(array, reference_metric) #comparison is fucking weird
                p_value_list.append(p)
                reference_metric_list.append(reference_metric)
                if estimate:
                    del c,d,X,Y,Z,_w,w,X_q

                else:
                    del c,X,Y,Z,_w,w,X_q

            p_value_array = torch.tensor(p_value_list)
            torch.save(p_value_array,
                       f'./{data_dir}/{job_dir}/p_val_array{suffix}.pt')
            ref_metric_array = torch.tensor(reference_metric_list)
            torch.save(ref_metric_array,
                       f'./{data_dir}/{job_dir}/ref_val_array{suffix}.pt')
            ks_stat, p_val_ks_test = kstest(p_value_array.numpy(), 'uniform')
            print(f'KS test Uniform distribution test statistic: {ks_stat}, p-value: {p_val_ks_test}')
            ks_data.append([ks_stat, p_val_ks_test])
            if estimator in ['NCE','TRE','linear_classifier'] and estimate:
                hsic_pval_list = torch.tensor(hsic_pval_list).float()
                r2_tensor = torch.tensor(R2_errors).float()
                torch.save(hsic_pval_list,
                           f'./{data_dir}/{job_dir}/hsic_pval_array{suffix}.pt')
                torch.save(r2_tensor,
                           f'./{data_dir}/{job_dir}/r2_array{suffix}.pt')
                df_data = torch.stack([hsic_pval_list,r2_tensor],dim=1).numpy()
                df_perf = pd.DataFrame(df_data,columns=['hsic_pval','r2'])
                df_perf.to_csv(f'./{data_dir}/{job_dir}/perf{suffix}.csv')
                s_perf = df_perf.describe()
                s_perf.to_csv(f'./{data_dir}/{job_dir}/psum{suffix}.csv')

        df = pd.DataFrame(ks_data, columns=['ks_stat', 'p_val_ks_test'])
        df.to_csv(f'./{data_dir}/{job_dir}/df{suffix}.csv')
        s = df.describe()
        s.to_csv(f'./{data_dir}/{job_dir}/summary{suffix}.csv')
        return

