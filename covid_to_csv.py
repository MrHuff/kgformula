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
    with open("covid_19_1/within_grouping.txt", "rb") as fp:
        index_list = pickle.load(fp)
    new_dirname_csv = 'covid_hdm_data'
    if not os.path.exists(new_dirname_csv):
        os.makedirs(new_dirname_csv)

    #think stratified
    for t_i in treatment_indices:
        treatment = treatments[t_i]
        X, Y, Z, ind_dict = torch.load(f'covid_19_1/data_treatment={treatment}.pt')
        for i in range(100): # Subset of countries
            print('seed: ',i)
            random.seed(i)
            country_subsets = sample_countries(index_list,num=25)
            x,y,z,index_list = transform_data(X,Y,Z,country_subsets)
            x_csv = x.cpu().numpy()
            y_csv = y.cpu().numpy()
            z_csv = z.cpu().numpy()
            np.savetxt(f"{new_dirname_csv}/x_{treatment}_{i}.csv", x_csv, delimiter=",")
            np.savetxt(f"{new_dirname_csv}/y_{treatment}_{i}.csv", y_csv, delimiter=",")
            np.savetxt(f"{new_dirname_csv}/z_{treatment}_{i}.csv", z_csv, delimiter=",")





