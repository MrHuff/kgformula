import os
import pandas as pd
import torch
from kgformula.utils import simulation_object_rule_new
import pickle
import random
import numpy as np
treatments = ['npi_school_closing', 'npi_workplace_closing', 'npi_cancel_public_events', 'npi_gatherings_restrictions', 'npi_close_public_transport', 'npi_stay_at_home', 'npi_internal_movement_restrictions', 'npi_international_travel_controls', 'npi_income_support', 'npi_debt_relief', 'npi_fiscal_measures', 'npi_international_support', 'npi_public_information', 'npi_testing_policy', 'npi_contact_tracing', 'npi_masks','auto_corr_ref']
treatment_indices = [0, 1, 2, 3, 4,-2, -1]
# treatment_indices = [ -1]

def sample_countries(index_list,num=50):
    indices = list(range(len(index_list)))
    sub_samples = random.sample(indices,num)
    country_subsets = [index_list[i] for i in sub_samples]
    return country_subsets

def transform_data(x,y,z,country_subsets):
    x_list = []
    y_list = []
    z_list = []
    reindexed_subsets = [0]
    for i,intervals in enumerate(country_subsets):
        diff = intervals[1] - intervals[0]
        x_sub = x[intervals[0]:intervals[1],:]
        y_sub = y[intervals[0]:intervals[1],:]
        z_sub = z[intervals[0]:intervals[1],:]
        x_list.append(x_sub)
        y_list.append(y_sub)
        z_list.append(z_sub)
        reindexed_subsets.append(diff)
    x_new =torch.cat(x_list,0)
    y_new =torch.cat(y_list,0)
    z_new =torch.cat(z_list,0)
    cum_sum = np.cumsum(reindexed_subsets)
    new_list = []
    for i in range(1,cum_sum.shape[0]):
        new_list.append([cum_sum[i-1],cum_sum[i]])
    return x_new,y_new,z_new,new_list



if __name__ == '__main__':
    with open("covid_19_1/within_grouping.txt", "rb") as fp:
        index_list = pickle.load(fp)

    for within_grouping in [True,False]:
        for t in treatment_indices:
            X,Y,Z,ind_dict = torch.load(f'covid_19_1/data_treatment={treatments[t]}.pt')
            n = X.shape[0]
            for m in [400]:
                for est in ['NCE_Q','real_TRE_Q']:
                    for sep in [False]:
                        for ts in [True]:
                            args = {'qdist':2,
                                    'bootstrap_runs':250,
                                    'est_params': {'lr': 1e-3, #use really small LR for TRE. Ok what the fuck is going on...
                                                                               'max_its': 50,
                                                                               'width': 32,
                                                                               'layers':3,
                                                                               'mixed': False,
                                                                               'bs_ratio': 1e-1,
                                                                               'val_rate': 0.1,
                                                                               'n_sample': 250,
                                                                               'criteria_limit': 0.05,
                                                                               'kill_counter': 2,
                                                                                'kappa':10 if est=='NCE_Q' else 1,
                                                                               'm': 4,
                                                                               'separate': sep
                                                   },
                                    'estimator': est,
                                    'cuda': True,
                                    'device': 'cuda:0',
                                    'n':10000,
                                    'unique_job_idx':0,
                                    'variant':1,
                                    }
                            pval_list = []
                            for i in range(100):
                                # i=21
                                if within_grouping:
                                    print('seed: ',i)
                                    random.seed(i)
                                    country_subsets = sample_countries(index_list)
                                    x,y,z,index_list = transform_data(X,Y,Z,country_subsets)
                                else:
                                    print('seed: ',i)
                                    torch.random.manual_seed(i)
                                    perm = torch.randperm(n)[:m]
                                    perm, indices = torch.sort(perm)
                                    x = X[perm]
                                    y = Y[perm]
                                    z = Z[perm]
                                    index_list=None
                                try:
                                    sim_obj = simulation_object_rule_new(args=args) #Why getting nans?
                                    pval, ref = sim_obj.run_data(x,y,z,ind_dict,time_series_data=ts,within_perm_vec=index_list)
                                except Exception as e:
                                    print (e)
                                pval_list.append(pval)
                            pvals = torch.tensor(pval_list)
                            torch.save({'pvals':pvals,'config':args},f'covid_pvals_ts={ts}_{within_grouping}_{est}_{sep}_{m}_{treatments[t]}_.pt')



