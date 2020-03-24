import argparse
import GPUtil
from matplotlib import pyplot as plt
from scipy.stats import kstest
import tqdm
import pandas as pd
import torch
from kgformula.test_statistics import weighted_stat,weighted_statistic_new,wild_bootstrap_deviance, density_estimator
from kgformula.fixed_do_samplers import simulate_xyz
import os
import numpy as np
from matplotlib.colors import ListedColormap
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def generate_data(y_a,y_b,z_a,z_b,cor,n,seeds):
    beta = {'y':[y_a,y_b],'z':[z_a,z_b]}
    if y_b == 0:
        ground_truth = 'H_0'
    else:
        ground_truth = 'H_1'
    data_dir = f'ground_truth={ground_truth}_y_a={y_a}_y_b={y_b}_z_a={z_a}_z_b={z_b}_cor={cor}_n={n}_seeds={seeds}'
    if not os.path.exists(f'./{data_dir}/'):
        os.mkdir(f'./{data_dir}/')
        for i in range(seeds):
            X, Y, Z, w = simulate_xyz(n=n, beta=beta, cor=cor, fam=1, oversamp=10,seed=i)
            torch.save((X,Y,Z,w),f'./{data_dir}/data_seed={i}.pt')



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
    parser.add_argument('--test_stat', type=int, nargs='?', default=1, help='runs')

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
    return pval

def get_mean_and_std(data):
    with torch.no_grad():
        mean = data.mean(dim=0)
        median,_ = data.median(dim=0)
        std = data.var(dim=0)**0.5
        x = [i for i in range(1,data.shape[1]+1)]
        return x, mean.cpu().numpy(),std.cpu().numpy(),median.cpu().numpy()


def run_job_func(args):
    j = simulation_object(args)
    j.run()

class simulation_object():
    def __init__(self,args):
        self.args=args

    def run(self):
        estimate = self.args['estimate']
        debug_plot = self.args['debug_plot']
        data_dir = self.args['data_dir']
        test_stat = self.args['test_stat']
        seeds = self.args['seeds']
        bootstrap_runs  = self.args['bootstrap_runs']
        bins = 25
        alpha = self.args['alpha']
        estimator = self.args['estimator']
        lamb = self.args['lamb']
        runs = self.args['runs']
        cuda = self.args['cuda']
        if cuda:
            device = GPUtil.getFirstAvailable(order='memory')[0]
        else:
            device = 'cpu'
        ks_data = []

        for j in range(runs):
            p_value_list = []
            reference_metric_list = []
            for i in tqdm.trange(seeds):
                if cuda:
                    X, Y, Z, w = torch.load(f'./{data_dir}/data_seed={i}.pt',map_location=f'cuda:{device}')
                else:
                    X, Y, Z, w = torch.load(f'./{data_dir}/data_seed={i}.pt')
                if debug_plot:
                    plt.scatter(Z.numpy(), X.numpy())
                    plt.show()
                # Cheating case
                if estimate:
                    d = density_estimator(x=X, z=Z, cuda=cuda, alpha=alpha, type=estimator, reg_lambda=lamb,device=device)
                    w = d.return_weights()
                if test_stat == 3:
                    c = weighted_statistic_new(X=X, Y=Y, Z=Z, w=w, cuda=cuda, device=device)
                elif test_stat == 2:
                    c = weighted_stat(X=X, Y=Y, Z=Z, w=w, cuda=cuda, device=device, half_mode=False)
                elif test_stat == 1:
                    c = wild_bootstrap_deviance(X=X, Y=Y, Z=Z, cuda=cuda, device=device)
                reference_metric = c.calculate_weighted_statistic()
                list_of_metrics = []
                for i in range(bootstrap_runs):
                    list_of_metrics.append(c.permutation_calculate_weighted_statistic())
                array = torch.tensor(list_of_metrics).float()
                p = calculate_pval(array, reference_metric)
                p_value_list.append(p.item())
                reference_metric_list.append(reference_metric.item())

            plt.hist(p_value_list, bins=bins)
            plt.savefig(
                f'./{data_dir}/p_value_plot_null=False_test_stat={test_stat}_seeds={seeds}_lambda={lamb}_estimate={estimate}_estimator={estimator}.png')
            plt.close()
            p_value_array = torch.tensor(p_value_list)
            torch.save(p_value_array,
                       f'./{data_dir}/null=False_p_value_array_seeds={seeds}_lambda={lamb}_estimate={estimate}_estimator={estimator}.pt')
            plt.hist((p_value_array + 1e-3).log().numpy(), bins=bins)
            plt.savefig(
                f'./{data_dir}/logp_value_plot_null=False_test_stat={test_stat}_seeds={seeds}_lambda={lamb}_estimate={estimate}_estimator={estimator}.png')
            plt.close()
            reference_metric_list = reject_outliers(reference_metric_list)
            plt.hist(reference_metric_list, bins=100)
            plt.savefig(f'./{data_dir}/ref_metric_plot_estimate={estimate}_estimator={estimator}_test_stat={test_stat}_lambda={lamb}_seeds={seeds}.png')
            plt.close()
            ks_stat, p_val_ks_test = kstest(p_value_array.numpy(), 'uniform')
            print(f'KS test Uniform distribution test statistic: {ks_stat}, p-value: {p_val_ks_test}')
            ks_data.append([ks_stat, p_val_ks_test])

        df = pd.DataFrame(ks_data, columns=['ks_stat', 'p_val_ks_test'])
        df.to_csv(f'./{data_dir}/df_{test_stat}_seeds={seeds}_lambda={lamb}_estimate={estimate}_estimator={estimator}.csv')
        s = df.describe()
        s.to_csv(f'./{data_dir}/summary_{test_stat}_seeds={seeds}_lambda={lamb}_estimate={estimate}_estimator={estimator}.csv')

    def plot_and_save(self,x, mean, std, median, name='xyz'):
        reg = self.args['lamb']
        data_dir = self.args['data_dir']
        alpha = self.args['alpha']

        plt.errorbar(x, mean, yerr=std)
        plt.savefig(f'./{data_dir}/{name}_{alpha}_{reg}.png')
        plt.clf()
        plt.scatter(x, median)
        plt.savefig(f'./{data_dir}/{name}_{alpha}_{reg}_median_scatter.png')
        plt.clf()

    def do_error_plot(self,data, name):
        x, mean, std, median = get_mean_and_std(data)
        self.plot_and_save(x, mean, std, median, name)

    def big_histogram(self,data,name):
        data_dir = self.args['data_dir']
        arr = data.flatten().cpu().numpy()
        plt.hist(arr,bins=self.args['seeds'])
        plt.savefig(f'./{data_dir}/{name}_histogram_big.png')
        plt.clf()

    def error_classification(self,errors):
        errors = errors.flatten().cpu().unsqueeze(-1).numpy()
        low = np.quantile(errors, 0.3)
        mid = np.quantile(errors, 0.7)
        mask_low = errors<low
        mask_mid = (errors>=low) & (errors<=mid)
        mask_high = errors> mid
        errors[mask_low] = 0
        errors[mask_mid] = 1
        errors[mask_high] = 2
        return errors
    def diagnostic_plot(self,errors, X, Z,name='xyz'):
        with torch.no_grad():
            reg = self.args['lamb']
            data_dir = self.args['data_dir']
            alpha = self.args['alpha']
            c = self.error_classification(errors)
            classes = ['low_error:green','medium_error:yellow','high_error:red']
            colours = ListedColormap(['g', 'y', 'r'])
            scatter = plt.scatter(X.cpu().numpy(), Z.cpu().numpy(),c=c,alpha=0.3,marker=".",cmap=colours)
            plt.legend(handles=scatter.legend_elements()[0], labels=classes)
            plt.savefig(f'./{data_dir}/{name}_{alpha}_{reg}_median_scatter.png')
            plt.clf()
    def debug_w(self,lambdas,expected_shape,estimator='truth'):
        data_dir = self.args['data_dir']
        seeds = self.args['seeds']
        cuda = self.args['cuda']
        if cuda:
            device = GPUtil.getFirstAvailable(order='memory')[0]
        else:
            device = 'cpu'
        for lamb in lambdas:
            self.args['lamb'] = lamb
            dmw_true = []
            dmw_kmm = []
            big_X = []
            big_Z = []
            for i in tqdm.trange(seeds):
                if cuda:
                    X, Y, Z, w = torch.load(f'./{data_dir}/data_seed={i}.pt',map_location=f'cuda:{device}')
                else:
                    X, Y, Z, w = torch.load(f'./{data_dir}/data_seed={i}.pt')
                if X.shape[0] != expected_shape:
                    print(X.shape[0])
                    continue
                else:
                    big_X.append(X)
                    big_Z.append(Z)
                    # Cheating case
                    dmw_true.append(w)
                    if estimator=='kmm':
                        d = density_estimator(x=X, z=Z, cuda=True, alpha=0, type='kmm', reg_lambda=lamb,device=device)
                        w_kmm = d.return_weights()
                        dmw_kmm.append(w_kmm)
            big_X  = torch.cat(big_X,dim=0)
            big_Z  = torch.cat(big_Z,dim=0)
            plot_true = torch.stack(dmw_true, dim=0)
            print(f'shape of plot_true = {plot_true.shape}')
            print(f'shape of big_X = {big_X.shape}')
            print(f'shape of big_Z = {big_Z.shape}')

            if estimator=='truth':
                self.do_error_plot(plot_true, 'w_true')
                self.big_histogram(plot_true,'w_true')
            elif estimator=='kmm':
                plot_kmm = torch.stack(dmw_kmm,dim=0)
                print(f'shape of plot_kmm = {plot_kmm.shape}')
                self.do_error_plot(plot_kmm, 'w_kmm')
                error_plot = torch.abs(plot_true-plot_kmm)
                self.do_error_plot(error_plot, 'error_w_kmm')
                self.diagnostic_plot(error_plot,big_X,big_Z,'diagnostic_plot')
                self.big_histogram(plot_kmm,f'w_kmm_lambda={lamb}')

