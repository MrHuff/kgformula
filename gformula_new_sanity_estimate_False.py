from kgformula.test_statistics import weighted_stat,weigted_statistic_new,wild_bootstrap_deviance, density_estimator
from kgformula.fixed_do_samplers import simulate_xyz
import os
import torch
import tqdm
from kgformula.utils import hypothesis_acceptance,calculate_pval
import GPUtil
from matplotlib import pyplot as plt
from scipy.stats import kstest
import pandas as pd
if __name__ == '__main__':
    #Log p-values, add vector of p-values
    beta = {'y':[0.0,0.5],'z':[0.0,0.5]}
    cor = 0.5
    data_dir = 'simulated_do_not_null_fixed_estimates_2'
    if not os.path.exists(f'./{data_dir}/'):
        os.mkdir(f'./{data_dir}/')
        seeds = 1000
        for i in range(seeds):
            X, Y, Z, w = simulate_xyz(n=1000, beta=beta, cor=cor, fam=1, oversamp=10,seed=i)
            torch.save((X,Y,Z,w),f'./{data_dir}/data_seed={i}.pt')
    plot = False
    test_stat = 2
    seeds = 1000
    bins = 25
    alpha = 0.0
    estimator = 'linear'
    device = GPUtil.getFirstAvailable()[0]
    runs = 5
    ks_data = []
    for j in range(runs):
        p_value_list = []
        for i in tqdm.trange(seeds):
            X,Y,Z,_ = torch.load(f'./{data_dir}/data_seed={i}.pt')
            if plot:
                plt.scatter(Z.numpy(),X.numpy())
                plt.show()
            #Cheating case
            d = density_estimator(numerator_sample=X,denominator_sample=Z,cuda=True,alpha=alpha,type=estimator)
            w = d.return_weights()
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
            p = calculate_pval(array, reference_metric)
            p_value_list.append(p.item())

        plt.hist(p_value_list, bins=bins)
        plt.savefig(f'./{data_dir}/p_value_plot_null=False_test_stat={test_stat}_seeds={seeds}_alpha={alpha}_estimator={estimator}.png')
        plt.close()
        p_value_array = torch.tensor(p_value_list)

        torch.save(p_value_array,f'./{data_dir}/null=False_p_value_array_seeds={seeds}_alpha={alpha}_estimator={estimator}.pt')
        plt.hist((p_value_array+1e-3).log().numpy(),bins=bins)
        plt.savefig(f'./{data_dir}/logp_value_plot_null=False_test_stat={test_stat}_seeds={seeds}_alpha={alpha}_estimator={estimator}.png')
        plt.close()
        ks_stat,p_val_ks_test = kstest(p_value_array.numpy(), 'uniform')
        print(f'KS test Uniform distribution test statistic: {ks_stat}, p-value: {p_val_ks_test}')
        ks_data.append([ks_stat,p_val_ks_test])

    df = pd.DataFrame(ks_data,columns=['ks_stat','p_val_ks_test'])
    s = df.describe()
    s.to_csv(f'./{data_dir}/summar_{test_stat}_seeds={seeds}_alpha={alpha}_estimator={estimator}.csv')