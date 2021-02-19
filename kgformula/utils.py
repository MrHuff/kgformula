import argparse
from matplotlib import pyplot as plt
import tqdm
import pandas as pd
import torch
from kgformula.test_statistics import *
from kgformula.fixed_do_samplers import apply_qdist
import os
import numpy as np
from scipy.stats import kstest
from matplotlib.colors import ListedColormap
import matplotlib.tri as mtri
import time
from kgformula.test_statistics import  hsic_sanity_check_w,hsic_test
import matplotlib as mpl
import os
mpl.use('Agg')
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
    d = density_estimator(x=X, z=Z, cuda=True, est_params=est_params, type=estimator,  device=device,secret_indx=1337,x_q=X)
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

class scale_dist():
    def __init__(self,X,q_fac):
        self.X = X
        self.q_fac = q_fac
    def sample(self,sizes):
        return self.X[:sizes[0],:]*self.q_fac

class x_q_class():
    def __init__(self,qdist,q_fac,X):
        self.q_fac = q_fac
        self.X = X
        self.theta = self.X.std(dim=0).squeeze()
        self.qdist = qdist
        if qdist == 1:
            self.q = Normal(self.X.mean(dim=0).squeeze(), self.q_fac * self.theta)
        elif qdist == 2:
            self.q  = scale_dist(X = self.X,q_fac=self.q_fac)
        elif qdist == 3:
            pass

    def sample(self,n):
        x_q = self.q.sample(torch.Size([n]))
        return x_q

    def calc_w_q(self,inv_wts):
        if self.qdist!=1:
            self.q = Normal(self.X.mean(dim=0).squeeze(), self.q_fac * self.theta)
            sample = self.q.sample(torch.Size([self.X.shape[0]]))
            # plt.hist(sample.numpy(), bins=50, alpha=0.5, color='b')
            # plt.hist(self.X.numpy(), bins=50, alpha=0.5, color='r')
            # plt.savefig('RIP.png')
            # plt.clf()
        q_dens = self.q.log_prob(self.X).sum(dim=1)
        q_dens = q_dens.exp()

        # plt.hist(q_dens.numpy(), bins=50, alpha=0.5, color='b')
        # plt.savefig('q_dens.png')
        # plt.clf()

        w_q = inv_wts * q_dens.squeeze()
        # plt.hist(w_q.numpy(), bins=50, alpha=0.5, color='b')
        # plt.savefig('RIP_2.png')
        # plt.clf()
        return w_q

    def calc_w_q_sanity_exp(self, inv_wts):
        if self.qdist != 1:
            self.q = Gamma(concentration=self.q_fac,rate=1./self.X.mean(dim=0).squeeze())
            # sample = self.q.sample(torch.Size([self.X.shape[0]]))
            # plt.hist(sample.numpy(),bins=50,alpha=0.5,color='b')
            # plt.hist(self.X.numpy(),bins=50,alpha=0.5,color='r')
            # plt.savefig('RIP_exp.png')
            # plt.clf()
        q_dens = self.q.log_prob(self.X).sum(dim=1)
        q_dens = q_dens.exp()

        # plt.hist(q_dens.numpy(), bins=50, alpha=0.5, color='b')
        # plt.savefig('q_dens_exp.png')
        # plt.clf()

        w_q = inv_wts * q_dens.squeeze()
        # plt.hist(w_q.numpy(), bins=50, alpha=0.5, color='b')
        # plt.savefig('RIP_2_exp.png')
        # plt.clf()
        return w_q


class simulation_object():
    def __init__(self,args):
        self.args=args
        self.cuda = self.args['cuda']
        self.device = self.args['device']
        self.validation_chunks = 10
        self.validation_over_samp = 10

    def validity_sanity_check(self,X_test,Y_test,Z_test,density_est):
        x_chunk = torch.chunk(X_test,self.validation_chunks)
        y_chunk = torch.chunk(Y_test,self.validation_chunks)
        z_chunk = torch.chunk(Z_test,self.validation_chunks)
        p_values = []
        for x,y,z in zip(x_chunk,y_chunk,z_chunk):
            x_keep,y_keep,z_keep,w_keep = self.validity_bootstrap_and_rejection_sampling(x,y,z,density_est)
            x_q_c = x_q_class(qdist=self.qdist, q_fac=self.q_fac, X=x_keep)
            x_q_keep = x_q_c.sample(x_keep.shape[0])
            p,ref_val = self.perm_Q_test(x_keep,y_keep,x_q_keep,w_keep,i=np.random.randint(0,1000))
            p_values.append(p)
        return p_values
    def perm_Q_test(self,X,Y,X_q,w,i):
        c = Q_weighted_HSIC(X=X, Y=Y, X_q=X_q, w=w, cuda=self.cuda, device=self.device, perm='Y', seed=i)
        reference_metric = c.calculate_weighted_statistic().cpu().item()
        list_of_metrics = []
        for i in range(self.bootstrap_runs):
            list_of_metrics.append(c.permutation_calculate_weighted_statistic().cpu().item())
        array = torch.tensor(
            list_of_metrics).float()  # seem to be extremely sensitive to lengthscale, i.e. could be sign flipper
        p = calculate_pval(array, reference_metric)  # comparison is fucking weird
        return p,reference_metric
    def validity_bootstrap_and_rejection_sampling(self,x,y,z,density_est):
        self.base_n = y.shape[0]
        index_list = list(range(self.base_n))
        bootstrap_samples = torch.tensor(np.random.choice(index_list,size=(self.base_n*self.validation_over_samp),replace=True)).long()
        bootstrap_y,bootstrap_z = y[bootstrap_samples,:],z[bootstrap_samples,:]
        bootstrap_x = x.repeat_interleave(self.validation_over_samp,dim=0)
        if bootstrap_x.dim()<2:
            bootstrap_x = bootstrap_x.unsqueeze(-1)
        with torch.no_grad():
            w = density_est.return_weights(bootstrap_x, bootstrap_z, bootstrap_x)
        w_rej = 1./w
        w_rej = w_rej/w_rej.max()
        r = torch.rand_like(w)
        keep = r<=w_rej
        x_keep,y_keep,z_keep,w_keep = bootstrap_x[keep,:],bootstrap_y[keep,:],bootstrap_z[keep,:],w[keep]
        return x_keep,y_keep,z_keep,w_keep

    def run(self):
        estimate = self.args['estimate']
        job_dir = self.args['job_dir']
        data_dir = self.args['data_dir']
        seeds_a = self.args['seeds_a']
        seeds_b = self.args['seeds_b']
        self.q_fac = self.args['q_factor']
        self.qdist = self.args['qdist']
        self.bootstrap_runs  = self.args['bootstrap_runs']
        est_params = self.args['est_params']
        estimator = self.args['estimator']
        mode = self.args['mode']
        split_data = self.args['split']
        required_n = self.args['n']
        exp_sanity = self.args['sanity_exp']
        ks_data = []
        R2_errors = []
        hsic_pval_list = []
        estimator_list = ['NCE', 'TRE_Q','NCE_Q', 'real_TRE_Q','rulsif']
        suffix = f'_qf={self.q_fac}_qd={self.qdist}_m={mode}_s={seeds_a}_{seeds_b}_e={estimate}_est={estimator}_sp={split_data}_br={self.bootstrap_runs}_n={required_n}'
        if not os.path.exists(f'./{data_dir}/{job_dir}'):
            os.makedirs(f'./{data_dir}/{job_dir}')
        mse_loss = torch.nn.MSELoss()

        if os.path.exists(f'./{data_dir}/{job_dir}/df{suffix}.csv'):
            return

        p_value_list = []
        reference_metric_list = []
        validity_p_list = []
        validity_stat_list = []
        actual_pvalues_validity = []
        for i in tqdm.trange(seeds_a,seeds_b):
            if self.cuda:
                X, Y, Z,_w = torch.load(f'./{data_dir}/data_seed={i}.pt',map_location=f'cuda:{self.device}')
            else:
                X, Y, Z,_w = torch.load(f'./{data_dir}/data_seed={i}.pt')

            X, Y, Z, _w = X[:required_n,:],Y[:required_n,:],Z[:required_n,:],_w[:required_n]
            Xq_class = x_q_class(qdist=self.qdist,q_fac=self.q_fac,X=X)
            X_q = Xq_class.sample(n=X.shape[0])
            if exp_sanity:
                w_q = Xq_class.calc_w_q_sanity_exp(_w)
            else:
                w_q = Xq_class.calc_w_q(_w)
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
                d = density_estimator(x=X_train, z=Z_train,x_q=X_q_train, cuda=self.cuda,
                                      est_params=est_params, type=estimator, device=self.device,secret_indx=self.args['unique_job_idx'])
                w = d.return_weights(X_test,Z_test,X_q_test)
                p_values_h_0 = self.validity_sanity_check(X_test, Y_test, Z_test, d)
                actual_pvalues_validity.append(torch.tensor(p_values_h_0))
                stat, pval =kstest(p_values_h_0,'uniform')
                validity_p_list.append(pval)
                validity_stat_list.append(stat)
                if estimator in estimator_list:
                    with torch.no_grad():
                        l = mse_loss(_w, w) / _w.var()
                    R2_errors.append(1-l.item())
                    hsic_pval_list.append(d.hsic_pval)
                X_q_test = d.X_q_test
                if i == 0:
                    torch.save(w,f'./{data_dir}/{job_dir}/w_estimated{suffix}.pt')
            else:
                w = w_q
            p,reference_metric = self.perm_Q_test(X_test,Y_test,X_q_test,w,i)
            p_value_list.append(p)
            reference_metric_list.append(reference_metric)

            if estimate:
                del d,X,Y,Z,_w,w,X_q
            else:
                del X,Y,Z,_w,w,X_q

        p_value_array = torch.tensor(p_value_list)
        torch.save(p_value_array,
                   f'./{data_dir}/{job_dir}/p_val_array{suffix}.pt')
        ref_metric_array = torch.tensor(reference_metric_list)
        torch.save(ref_metric_array,
                   f'./{data_dir}/{job_dir}/ref_val_array{suffix}.pt')
        ks_stat, p_val_ks_test = kstest(p_value_array.numpy(), 'uniform')
        print(f'KS test Uniform distribution test statistic: {ks_stat}, p-value: {p_val_ks_test}')
        ks_data.append([ks_stat, p_val_ks_test])
        if estimator in estimator_list and estimate:
            validity_p_value_array = torch.tensor(validity_p_list)
            validity_value_array = torch.tensor(validity_stat_list)
            hsic_pval_list = torch.tensor(hsic_pval_list).float()
            r2_tensor = torch.tensor(R2_errors).float()
            actual_pvalues_validity = torch.cat(actual_pvalues_validity).float()
            torch.save(actual_pvalues_validity,
                       f'./{data_dir}/{job_dir}/actual_validity_p_value_array{suffix}.pt')
            torch.save(validity_p_value_array,
                       f'./{data_dir}/{job_dir}/validity_p_value_array{suffix}.pt')
            torch.save(validity_value_array,
                       f'./{data_dir}/{job_dir}/validity_value_array{suffix}.pt')
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

class simulation_object_hsic():
    def __init__(self,args):
        self.args=args
        self.cuda = self.args['cuda']
        self.device = self.args['device']

    def run(self):
        job_dir = self.args['job_dir']
        data_dir = self.args['data_dir']
        seeds_a = self.args['seeds_a']
        seeds_b = self.args['seeds_b']
        self.q_fac = self.args['q_factor']
        self.qdist = self.args['qdist']
        self.bootstrap_runs = self.args['bootstrap_runs']
        required_n = self.args['n']
        suffix = f'_hsic_s={seeds_a}_{seeds_b}_br={self.bootstrap_runs}_n={required_n}'
        if not os.path.exists(f'./{data_dir}/{job_dir}'):
            os.makedirs(f'./{data_dir}/{job_dir}')
        if os.path.exists(f'./{data_dir}/{job_dir}/df{suffix}.csv'):
            return
        p_value_list = []
        ks_data = []
        for i in tqdm.trange(seeds_a,seeds_b):
            if self.cuda:
                X, Y, Z,_w = torch.load(f'./{data_dir}/data_seed={i}.pt',map_location=f'cuda:{self.device}')
            else:
                X, Y, Z,_w = torch.load(f'./{data_dir}/data_seed={i}.pt')

            p_val = hsic_test(X,Y,n_sample=self.bootstrap_runs)
            p_value_list.append(p_val)
        p_value_array = torch.tensor(p_value_list)
        torch.save(p_value_array,
                   f'./{data_dir}/{job_dir}/p_val_array{suffix}.pt')
        ks_stat, p_val_ks_test = kstest(p_value_array.numpy(), 'uniform')
        print(f'KS test Uniform distribution test statistic: {ks_stat}, p-value: {p_val_ks_test}')
        ks_data.append([ks_stat, p_val_ks_test])
        df = pd.DataFrame(ks_data, columns=['ks_stat', 'p_val_ks_test'])
        df.to_csv(f'./{data_dir}/{job_dir}/df{suffix}.csv')
        s = df.describe()
        s.to_csv(f'./{data_dir}/{job_dir}/summary{suffix}.csv')
        return
