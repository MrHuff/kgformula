from kgformula.test_statistics import weighted_stat,weigted_statistic_new,wild_bootstrap_deviance
from kgformula.fixed_do_samplers import simulate_xyz
import os
import torch
import tqdm
from kgformula.utils import hypothesis_acceptance,calculate_power
import GPUtil
from matplotlib import pyplot as plt
from scipy.stats import kstest

if __name__ == '__main__':
    #Log p-values, add vector of p-values
    beta = {'y':[0,0],'z':[0,0.5]}
    cor = 0.5
    data_dir = 'simulated_do_null_fixed'
    if not os.path.exists(f'./{data_dir}/'):
        os.mkdir(f'./{data_dir}/')
        seeds = 1000
        for i in range(seeds):
            X, Y, Z, w = simulate_xyz(n=1000, beta=beta, cor=cor, fam=1, oversamp=10,seed=i)
            torch.save((X,Y,Z,w),f'./{data_dir}/data_seed={i}.pt')
    else:
        plot = False
        test_stat = 2
        seeds = 1000
        bins = 25
        p_value_list = []
        device = GPUtil.getFirstAvailable()[0]
        for i in tqdm.trange(seeds):
            X,Y,Z,w = torch.load(f'./{data_dir}/data_seed={i}.pt')
            if plot:
                plt.scatter(Z.numpy(),X.numpy())
                plt.show()
            #Cheating case
            if test_stat == 3:
                c = weigted_statistic_new(X=X, Y=Y, Z=Z, w=w, cuda=True, device=device)
            elif test_stat == 2:
                c = weighted_stat(X=X,Y=Y,Z=Z,w=w,cuda=True,device=device,half_mode=False)
            elif test_stat == 1:
                c = wild_bootstrap_deviance(X=X,Y=Y,Z=Z,cuda=True,device=device)
            reference_metric = c.calculate_weighted_statistic()
            list_of_metrics = []
            for i in range(250):
                list_of_metrics.append(c.permutation_calculate_weighted_statistic())
            array = torch.tensor(list_of_metrics).float()
            p = calculate_power(array,reference_metric)
            p_value_list.append(p.item())

        plt.hist(p_value_list, bins=bins)
        plt.savefig(f'./{data_dir}/p_value_plot_null=True_test_stat={test_stat}_seeds={seeds}.png')
        plt.close()
        p_value_array = torch.tensor(p_value_list)

        torch.save(p_value_array,f'./{data_dir}/p_value_array_seeds={seeds}.pt')
        plt.hist((p_value_array+1e-3).log().numpy(),bins=bins)
        plt.savefig(f'./{data_dir}/logp_value_plot_null=True_test_stat={test_stat}_seeds={seeds}.png')
        ks_stat,p_val_ks_test = kstest(p_value_array.numpy(), 'uniform')
        print(f'KS test Uniform distribution test statistic: {ks_stat}, p-value: {p_val_ks_test}')