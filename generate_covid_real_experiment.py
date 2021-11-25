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

def save_obj(obj, name ,folder):
    with open(f'{folder}'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name,folder):
    with open(f'{folder}' + name, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    JOB_NAME = 'weekly_covid_within_n_blocks_50_y'
    dataset = 'covid_19_2'
    if not os.path.exists(JOB_NAME):
        os.makedirs(JOB_NAME)
    counter = 0
    for n_blocks in [2,3,4]:
        for within_grouping in [True]:
            for t in treatment_indices:
                for m in [2500]:
                    for est in ['NCE_Q','real_TRE_Q']:
                        for sep in [False]:
                            for ts in [True]:
                                args = {'qdist':2,
                                        'bootstrap_runs':250,
                                        'est_params': {'lr': 1e-3, #use really small LR for TRE. Ok what the fuck is going on...
                                                                                   'max_its': 50,
                                                                                   'width': 32,
                                                                                   'layers':4,
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
                                        'n':m,
                                        'unique_job_idx':0,
                                        'variant':1,
                                        'ts':ts,
                                        'within_grouping':within_grouping,
                                        'treatment': treatments[t],
                                        'dir_name_jobs': JOB_NAME,
                                        'n_blocks':n_blocks,
                                        'dataset_covid':dataset
                                        }

                                save_obj(args,f'job_{counter}',JOB_NAME+'/')
                                counter+=1



