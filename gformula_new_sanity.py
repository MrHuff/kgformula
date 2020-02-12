from kgformula.test_statistics import weighted_stat
from kgformula.fixed_do_samplers import simulate_xyz
import os
import torch
from kgformula.utils import hypothesis_acceptance,calculate_power
from matplotlib import pyplot as plt
if __name__ == '__main__':
    beta = {'y':[0,0],'z':[0,0.5]}
    cor = 0.5
    if not os.path.exists('./simulated_do_null_fixed/'):
        os.mkdir('./simulated_do_null_fixed/')
        seeds = 1000
        for i in range(seeds):
            X, Y, Z, w = simulate_xyz(n=1000, beta=beta, cor=cor, fam=1, oversamp=2)
            torch.save((X,Y,Z,w),f'./simulated_do_null_fixed/data_seed={i}.pt')
    else:
        seeds = 100
        p_value_list = []
        for i in range(seeds):
            X,Y,Z,w = torch.load(f'./simulated_do_null_fixed/data_seed={i}.pt')
            #Cheating case
            c = weighted_stat(X=X,Y=Y,Z=Z,w=w)
            reference_metric = c.calculate_weighted_statistic()
            list_of_metrics = []
            for i in range(100):
                list_of_metrics.append(c.permutation_calculate_weighted_statistic())
            array = torch.tensor(list_of_metrics)
            p = calculate_power(array,reference_metric)
            p_value_list.append(p.item())
        plt.hist(p_value_list,bins=seeds//10)
        plt.savefig('./simulated_do_null_fixed/p_value_plot_null=True.png')