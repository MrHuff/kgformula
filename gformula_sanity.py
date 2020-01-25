from kgformula.test_statistics import weighted_stat
from kgformula.do_samplers import do_distribution
import os
import torch
if __name__ == '__main__':

    if not os.path.exists('./simluated_do_null/'):
        os.mkdir('./simluated_do_null/')
        seeds = 100
        for i in range(seeds):
            X, Y, Z, w = do_distribution(n=100, get_p_x_cond_z=True)
            torch.save((X,Y,Z,w),f'./simluated_do_null/data_seed={i}.pt')
    else:
        seed = 1
        X,Y,Z,w = torch.load(f'./simluated_do_null/data_seed={seed}.pt')
    #Cheating case
    # c = weighted_stat(X=X,Y=Y,Z=Z,get_p_x_cond_z=w)
    # reference_metric = c.calculate_weighted_statistic()
    # list_of_metrics = []
    # print(reference_metric)
    # for i in range(100):
    #     list_of_metrics.append(c.permutation_calculate_weighted_statistic())
    # print(list_of_metrics)
