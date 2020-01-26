from kgformula.test_statistics import weighted_stat
from kgformula.do_samplers import do_distribution
import os
import torch
from kgformula.utils import hypothesis_acceptance,calculate_power
from matplotlib import pyplot as plt
if __name__ == '__main__':

    if not os.path.exists('./simluated_do_null/'):
        os.mkdir('./simluated_do_null/')
        seeds = 100
        for i in range(seeds):
            X, Y, Z, w = do_distribution(n=100, get_p_x_cond_z=True)
            torch.save((X,Y,Z,w),f'./simluated_do_null/data_seed={i}.pt')
    else:
        seeds = 100
        p_value_list = []
        for i in range(seeds):
            X,Y,Z,w = torch.load(f'./simluated_do_null/data_seed={i}.pt')
            #Cheating case
            c = weighted_stat(X=X,Y=Y,Z=Z,get_p_x_cond_z=w)
            reference_metric = c.calculate_weighted_statistic()
            list_of_metrics = []
            for i in range(100):
                list_of_metrics.append(c.permutation_calculate_weighted_statistic())
            array = torch.tensor(list_of_metrics)
            p = calculate_power(array,reference_metric)
            p_value_list.append(p.item())
        plt.hist(p_value_list,bins=seeds//10)
        plt.savefig('./p_value_plot_null=True.png')