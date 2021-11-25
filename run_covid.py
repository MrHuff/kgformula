import os
import pandas as pd
import torch
from kgformula.utils import simulation_object_rule_new
import pickle
import random
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--idx', type=int, default=0, help='cdim')
parser.add_argument('--ngpu', type=int, default=4, help='cdim')
parser.add_argument('--job_folder', type=str, default='', help='cdim')
parser.add_argument('--num_cont', type=int, default=0, help='cdim')

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


    input  = vars(parser.parse_args())
    counter = input['idx']
    ngpu = input['ngpu']
    FOLD_LOAD = input['job_folder']
    num_countries = input['num_cont']
    args = load_obj(f'job_{counter}.pkl',FOLD_LOAD+'/')
    dataset_covid = args['dataset_covid']
    with open(f"{dataset_covid}/within_grouping.txt", "rb") as fp:
        index_list = pickle.load(fp)
    args['device'] = 0
    args['unique_job_idx'] = counter%ngpu
    n_blocks = args['n_blocks']
    JOB_SAVE_DIR = args['dir_name_jobs']+'_res'
    if counter==0:
        if not os.path.exists(JOB_SAVE_DIR):
            os.makedirs(JOB_SAVE_DIR)
    n = args['n']
    ts= args['ts']
    within_grouping = args['within_grouping']
    treatment = args['treatment']
    X, Y, Z, ind_dict = torch.load(f'{dataset_covid}/data_treatment={treatment}.pt')
    pval_list=[]
    sep=False
    est= args['estimator']

    #think stratified
    for i in range(100): # Subset of countries
        if args['within_grouping']:
            print('seed: ',i)
            random.seed(i)
            country_subsets = sample_countries(index_list,num=num_countries)
            x,y,z,index_list = transform_data(X,Y,Z,country_subsets)
        else:
            print('seed: ',i) #DONT DO THIS!
            torch.random.manual_seed(i)
            perm = torch.randperm(n)[:n]
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
    torch.save({'pvals':pvals,'config':args},f'{JOB_SAVE_DIR}/covid_pvals_ts={ts}_{within_grouping}_{est}_{sep}_{n}_{treatment}_nblocks={n_blocks}_.pt')



