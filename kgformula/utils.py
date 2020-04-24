import argparse
import GPUtil
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from scipy.stats import kstest
import tqdm
import pandas as pd
import torch
from kgformula.test_statistics import weighted_statistic_new, density_estimator
from kgformula.fixed_do_samplers import simulate_xyz
import os
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.tri as mtri
def get_density_plot(d,X,Z):
    w = d.return_weights()
    X = X.cpu().flatten().numpy()
    Z = Z.cpu().flatten().numpy()
    w = w.cpu().flatten().numpy()
    triang = mtri.Triangulation(X, Z)
    plt.tricontourf(triang, w)
    plt.colorbar()
    plt.show()
    plt.clf()
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

class simulation_object():
    def __init__(self,args):
        self.args=args
        self.cuda = self.args['cuda']
        if self.cuda:
            self.device = GPUtil.getFirstAvailable(order='memory')[0]
        else:
            self.device = 'cpu'

    def run(self):
        estimate = self.args['estimate']
        debug_plot = self.args['debug_plot']
        data_dir = self.args['data_dir']
        seeds = self.args['seeds']
        bootstrap_runs  = self.args['bootstrap_runs']
        bins = 25
        est_params = self.args['est_params']
        estimator = self.args['estimator']
        lamb = self.args['lamb']
        runs = self.args['runs']
        ks_data = []
        suffix = f'_seeds={seeds}_estimate={estimate}_estimator={estimator}'
        if estimator=='kmm':
            suffix = suffix + f'_{lamb}'
        elif estimator=='classifier':
            for key,val in est_params.items():
                suffix = suffix + f'_{key}={val}'

        for j in range(runs):
            p_value_list = []
            reference_metric_list = []
            for i in tqdm.trange(seeds):
                if self.cuda:
                    X, Y, Z, _w = torch.load(f'./{data_dir}/data_seed={i}.pt',map_location=f'cuda:{self.device}')
                else:
                    X, Y, Z, _w = torch.load(f'./{data_dir}/data_seed={i}.pt')
                if debug_plot:
                    plt.scatter(Z.numpy(), X.numpy())
                    plt.show()
                if estimate:
                    d = density_estimator(x=X, z=Z, cuda=self.cuda, est_params=est_params, type=estimator,device=self.device)
                    if d.failed:
                        continue
                    w = d.return_weights()
                else:
                    w = _w
                c = weighted_statistic_new(X=X, Y=Y, Z=Z, w=w, cuda=self.cuda, device=self.device)
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
                f'./{data_dir}/p_value_plot{suffix}.png')
            plt.close()
            p_value_array = torch.tensor(p_value_list)
            torch.save(p_value_array,
                       f'./{data_dir}/p_val_array{suffix}.pt')
            plt.hist((p_value_array + 1e-3).log().numpy(), bins=bins)
            plt.savefig(
                f'./{data_dir}/logp_value_plot{suffix}.png')
            plt.close()
            reference_metric_list = reject_outliers(reference_metric_list)
            plt.hist(reference_metric_list, bins=100)
            plt.savefig(f'./{data_dir}/ref_metric_plot{suffix}.png')
            plt.close()
            ks_stat, p_val_ks_test = kstest(p_value_array.numpy(), 'uniform')
            print(f'KS test Uniform distribution test statistic: {ks_stat}, p-value: {p_val_ks_test}')
            ks_data.append([ks_stat, p_val_ks_test])

        df = pd.DataFrame(ks_data, columns=['ks_stat', 'p_val_ks_test'])
        df.to_csv(f'./{data_dir}/df{suffix}.csv')
        s = df.describe()
        s.to_csv(f'./{data_dir}/summary{suffix}.csv')

    def plot_and_save(self,x, mean, std, median, name='xyz'):
        reg = self.args['lamb']
        data_dir = self.args['data_dir']
        plt.plot(x, mean,marker=".")
        plt.savefig(f'./{data_dir}/{name}_{reg}_mean_scatter.png')
        plt.clf()

        plt.scatter(x, median)
        plt.savefig(f'./{data_dir}/{name}_{reg}_median_scatter.png')
        plt.clf()

    def do_error_plot(self,data, name):
        x, mean, std, median = get_median_and_std(data)
        self.plot_and_save(x, mean, std, median, name)

    def big_histogram(self,data,name):
        data_dir = self.args['data_dir']
        arr = data.flatten().cpu().numpy()
        # arr = reject_outliers(arr,m=1)
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
            c = self.error_classification(errors)
            classes = ['low_error:green','medium_error:yellow','high_error:red']
            colours = ListedColormap(['g', 'y', 'r'])
            scatter = plt.scatter(X.cpu().numpy(), Z.cpu().numpy(),c=c,alpha=0.3,marker=".",cmap=colours)
            plt.legend(handles=scatter.legend_elements()[0], labels=classes)
            plt.savefig(f'./{data_dir}/{name}_{reg}_median_scatter.png')
            plt.clf()

    def estimator_error_plot(self,org,est,name):
        reg = self.args['lamb']
        data_dir = self.args['data_dir']
        o = np.array(org)
        e = np.array(est)
        error = (e-o)/o
        plt.hist(error,bins=50)
        plt.savefig(f'./{data_dir}/{name}_{reg}_expectation_error.png')
        plt.clf()

    def surface_plot(self,X,Z,errors,name=''):
        reg = self.args['lamb']
        data_dir = self.args['data_dir']
        errors = errors.flatten().cpu().numpy()
        ax = plt.axes(projection='3d')
        ax.plot_trisurf(X.cpu().flatten().numpy(), Z.cpu().flatten().numpy(), errors, cmap='viridis', edgecolor='none')
        ax.set_title('Surface plot')
        plt.savefig(f'./{data_dir}/{name}_{reg}_surface.png')
        plt.clf()

    def diagnose_technique(self,dmw,plot_true,big_X,big_Z,lamb,estimator,org,est):
        plot_technique = torch.stack(dmw, dim=0)
        print(f'shape of plot_{estimator} = {plot_technique.shape}')
        self.do_error_plot(plot_technique, f'w_{estimator}')
        error_plot = (plot_technique-plot_true)/plot_true
        self.do_error_plot(error_plot, f'error_w_{estimator}')
        self.diagnostic_plot(error_plot, big_X, big_Z, f'diagnostic_plot_{estimator}')
        self.big_histogram(plot_technique, f'w_{estimator}_lambda={lamb}')
        self.surface_plot(big_X,big_Z,error_plot,f'surface_plot_{estimator}')
        self.estimator_error_plot(org,est,f'{estimator}')

    def debug_w(self,lambdas,expected_shape,estimator='truth'):
        data_dir = self.args['data_dir']
        seeds = self.args['seeds']

        for lamb in lambdas:
            self.args['lamb'] = lamb
            dmw_true = []
            dmw_estimator = []
            big_X = []
            big_Z = []
            og_stat_list = []
            est_stat_list = []
            for i in tqdm.trange(seeds):
                if self.cuda:
                    X, Y, Z, w = torch.load(f'./{data_dir}/data_seed={i}.pt',map_location=f'cuda:{self.device}')
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
                        d = density_estimator(x=X, z=Z, cuda=True, est_params=None, type='kmm', reg_lambda=lamb,device=self.device)
                        w_estimator = d.return_weights()
                        dmw_estimator.append(w_estimator)
                    elif estimator=='semi':
                        d = density_estimator(x=X, z=Z, cuda=True, est_params=None, type='semi', reg_lambda=lamb,device=self.device)
                        w_estimator = d.return_weights()
                        dmw_estimator.append(w_estimator)
                    elif estimator=='classifier':
                        d = density_estimator(x=X, z=Z, cuda=True, est_params=self.args['est_params'], type='classifier', reg_lambda=lamb,
                                              device=self.device)
                        w_estimator = d.return_weights()
                        dmw_estimator.append(w_estimator)

                    c_0 = weighted_statistic_new(X=X, Y=Y, Z=Z, w=w, cuda=self.cuda, device=self.device)
                    orginal = c_0.calculate_weighted_statistic()
                    og_stat_list.append(orginal.item())

                    if estimator is not 'truth':
                        c_1 = weighted_statistic_new(X=X, Y=Y, Z=Z, w=w_estimator, cuda=self.cuda, device=self.device)
                        est = c_1.calculate_weighted_statistic()
                        est_stat_list.append(est.item())

            big_X  = torch.cat(big_X,dim=0)
            big_Z  = torch.cat(big_Z,dim=0)
            plot_true = torch.stack(dmw_true, dim=0)
            print(f'shape of plot_true = {plot_true.shape}')
            print(f'shape of big_X = {big_X.shape}')
            print(f'shape of big_Z = {big_Z.shape}')

            if estimator=='truth':
                self.do_error_plot(plot_true, 'w_true')
                self.big_histogram(plot_true,'w_true')
            else:
                self.diagnose_technique(dmw=dmw_estimator,
                                        plot_true=plot_true,
                                        big_X=big_X,
                                        big_Z=big_Z,
                                        lamb=lamb,
                                        estimator=estimator,
                                        org=og_stat_list,
                                        est=est_stat_list)

