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

def plot_and_save(x,mean,std,median,data_dir,name='xyz',alpha=0,reg=0):
    plt.errorbar(x,mean,yerr=std)
    plt.savefig(f'./{data_dir}/{name}_{alpha}_{reg}.png')
    plt.clf()
    plt.scatter(x,median)
    plt.savefig(f'./{data_dir}/{name}_{alpha}_{reg}_median_scatter.png')
    plt.clf()

def do_error_plot(data,data_dir,name,alpha,reg):
    x, mean,std,median = get_mean_and_std(data)
    plot_and_save(x,mean,std,median,data_dir,name,alpha,reg)

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
        device = GPUtil.getFirstAvailable(order='memory')[0]
        runs = self.args['runs']
        ks_data = []

        for j in range(runs):
            p_value_list = []
            reference_metric_list = []
            for i in tqdm.trange(seeds):
                X, Y, Z, w = torch.load(f'./{data_dir}/data_seed={i}.pt')
                if debug_plot:
                    plt.scatter(Z.numpy(), X.numpy())
                    plt.show()
                # Cheating case
                if estimate:
                    d = density_estimator(x=X, z=Z, cuda=True, alpha=alpha, type=estimator, reg_lambda=lamb)
                    w = d.return_weights()
                if test_stat == 3:
                    c = weighted_statistic_new(X=X, Y=Y, Z=Z, w=w, cuda=True, device=device)
                elif test_stat == 2:
                    c = weighted_stat(X=X, Y=Y, Z=Z, w=w, cuda=True, device=device, half_mode=False)
                elif test_stat == 1:
                    c = wild_bootstrap_deviance(X=X, Y=Y, Z=Z, cuda=True, device=device)
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
                f'./{data_dir}/p_value_plot_null=False_test_stat={test_stat}_seeds={seeds}_alpha={alpha}_estimator={estimator}.png')
            plt.close()
            p_value_array = torch.tensor(p_value_list)
            torch.save(p_value_array,
                       f'./{data_dir}/null=False_p_value_array_seeds={seeds}_alpha={alpha}_estimator={estimator}.pt')
            plt.hist((p_value_array + 1e-3).log().numpy(), bins=bins)
            plt.savefig(
                f'./{data_dir}/logp_value_plot_null=False_test_stat={test_stat}_seeds={seeds}_alpha={alpha}_estimator={estimator}.png')
            plt.close()
            plt.hist(reference_metric_list, bins=100)
            plt.savefig(f'./{data_dir}/ref_metric_plot_null=False_test_stat={test_stat}_seeds={seeds}.png')
            plt.close()
            ks_stat, p_val_ks_test = kstest(p_value_array.numpy(), 'uniform')
            print(f'KS test Uniform distribution test statistic: {ks_stat}, p-value: {p_val_ks_test}')
            ks_data.append([ks_stat, p_val_ks_test])

        df = pd.DataFrame(ks_data, columns=['ks_stat', 'p_val_ks_test'])
        df.to_csv(f'./{data_dir}/df_{test_stat}_seeds={seeds}_alpha={alpha}_estimator={estimator}.csv')
        s = df.describe()
        s.to_csv(f'./{data_dir}/summary_{test_stat}_seeds={seeds}_alpha={alpha}_estimator={estimator}.csv')

    # def debug_w(self):
    #     test_stat = 2
    #     n = 1000
    #     seeds = 1000
    #     bins = 25
    #     device = GPUtil.getFirstAvailable()[0]
    #     runs = 1
    #     ks_data = []
    #     for alpha in [0, 0.5, 1]:
    #         for lamb in [1e-3, 1e-2, 1e-1]:
    #             dmw_true = []
    #             dmw_semi = []
    #             dmw_gp = []
    #             dmw_linear = []
    #             for i in tqdm.trange(seeds):
    #                 X, Y, Z, w_true = torch.load(f'./{data_dir}/data_seed={i}.pt')
    #                 if X.shape[0] != n:
    #                     print(X.shape[0])
    #                     continue
    #                 else:
    #                     # Cheating case
    #                     # d = density_estimator(numerator_sample=X,denominator_sample=Z,cuda=True,alpha=alpha,type='semi',reg_lambda=lamb)
    #                     # w_semi = d.return_weights()
    #                     d = density_estimator(x=X, z=Z, cuda=True, alpha=alpha, type='gp', reg_lambda=lamb)
    #                     w_gp = d.return_weights()
    #                     d = density_estimator(x=X, z=Z, cuda=True, alpha=alpha, type='linear', reg_lambda=lamb)
    #                     w_linear = d.return_weights()
    #                     # dmw_true.append(w_true)
    #                     # dmw_semi.append(w_semi)
    #                     dmw_gp.append(w_gp)
    #                     dmw_linear.append(w_linear)
    #             # plot_true = torch.stack(dmw_true,dim=0)
    #             # plot_semi = torch.stack(dmw_semi,dim=0)
    #             plot_gp = torch.stack(dmw_gp, dim=0)
    #             plot_linear = torch.stack(dmw_linear, dim=0)
    #             # do_error_plot(plot_true,data_dir,'w_true',alpha=alpha,reg=lamb)
    #             # do_error_plot(plot_semi,data_dir,'w_semi',alpha=alpha,reg=lamb)
    #             do_error_plot(plot_gp, data_dir, 'w_gp', alpha=alpha, reg=lamb)
    #             do_error_plot(plot_linear, data_dir, 'w_linear', alpha=alpha, reg=lamb)

