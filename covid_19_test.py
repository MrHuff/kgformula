import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
import os
from sklearn.impute import KNNImputer
import random
import numpy as np
import pickle
pd.options.display.max_rows = 10000
pd.options.display.max_columns = 1000
columns_of_interest = ['DATE', 'country_name', 'npi_school_closing', 'npi_workplace_closing', 'npi_cancel_public_events', 'npi_gatherings_restrictions', 'npi_close_public_transport', 'npi_stay_at_home', 'npi_internal_movement_restrictions', 'npi_international_travel_controls', 'npi_income_support', 'npi_debt_relief', 'npi_fiscal_measures', 'npi_international_support', 'npi_public_information', 'npi_testing_policy', 'npi_contact_tracing', 'npi_healthcare_investment', 'npi_vaccine_investment', 'npi_masks', 'cases_total', 'cases_new', 'deaths_total', 'deaths_new', 'cases_total_per_million', 'cases_new_per_million', 'deaths_total_per_million', 'deaths_new_per_million', 'tests_total', 'tests_new', 'tests_total_per_thousand', 'tests_new_per_thousand', 'stats_population', 'stats_population_density', 'stats_median_age', 'stats_gdp_per_capita', 'cases_days_since_first', 'deaths_days_since_first', 'stats_hospital_beds_per_1000', 'stats_smoking', 'stats_population_urban', 'stats_population_school_age', 'deaths_excess_daily_avg', 'deaths_excess_weekly']
print(len(columns_of_interest))


def cat_fix(df):
    df = df.astype("category")
    return pd.get_dummies(df,drop_first=True)

def normalize(df):
    col_names = df.columns
    x = df.values
    s = StandardScaler()
    x = s.fit_transform(x)
    return pd.DataFrame(x,columns=col_names.tolist())

def filter_nan_countries(df):
    list = df['country_name'].unique().tolist()
    ret_list = []
    for el in list:
        subset = df[df['country_name']==el]
        if not subset.isnull().values.any():
            ret_list.append(el)
    return ret_list

def save_torch(X_pd,Y_pd,Z_cat,Z_cont,dir,filename):
    X = torch.from_numpy(X_pd.values).float()
    Y = torch.from_numpy(Y_pd.values).float()
    Z = torch.cat([torch.from_numpy(Z_cat.values).float(),torch.from_numpy(Z_cont.values).float()],dim=1)
    print(X.shape)
    print(Y.shape)
    print(Z.shape)
    col_stats_list=[]
    for i in range(Z_cat.shape[1]):
        col_stats = Z_cat.iloc[:,i].unique().tolist()
        col_stats_list.append(col_stats)
    w={ 'indicator':[True]*Z_cat.shape[1]+Z_cont.shape[1]*[False],'index_lists':col_stats_list}
    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save((X,Y,Z,w),dir+filename)

def save_torch_mult(X_pd,Y_pd,Z_cont,dir,filename):
    X = torch.from_numpy(X_pd.values).float()
    Y = torch.from_numpy(Y_pd.values).float()
    Z = torch.from_numpy(Z_cont.values).float()
    print(X.shape)
    print(Y.shape)
    print(Z.shape)
    col_stats_list=[]
    w={ 'indicator':Z_cont.shape[1]*[False],'index_lists':col_stats_list
}
    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save((X,Y,Z,w),dir+filename)

def shuffle_by_country(df):
    groups = [df for _, df in df.groupby('country_name')]
    random.shuffle(groups)
    df = pd.concat(groups).reset_index(drop=True)

    return df

def filter_quarter(df):
    indices = []
    for i,row in df.iterrows():
        month = row['DATE'].month
        day = row['DATE'].day
        if day==1 and month in [1,4,7,10]:
            indices.append(i)
    return df.iloc[indices,:]

def filter_week(df):
    indices = []
    for i,row in df.iterrows():
        # month = row['DATE'].month
        day = row['DATE'].day
        if day in  [1,7,14,21,28]:
            indices.append(i)
    return df.iloc[indices,:]

def generate_autocorrelated_treatment(df):
    n = df.shape[0]
    auto_cor = 0.5
    arr = [np.random.rand(1)]
    for i in range(1,n+1):
        arr.append( arr[i-1]*auto_cor + np.random.randn(1))
    ts = np.array(arr)
    df['auto_corr_ref'] = ts[1:]
    return df

if __name__ == '__main__':
    if not os.path.exists('covid_19_1'):
        os.makedirs('covid_19_1')

    df = pd.read_csv(
        'https://raw.githubusercontent.com/rs-delve/covid19_datasets/master/dataset/combined_dataset_latest.csv',
        parse_dates=['DATE'])

    df = df[columns_of_interest]
    df = shuffle_by_country(df)


    cols_1 = df.columns.tolist()

    # df = filter_quarter(df)
    df = filter_week(df)


    df = df.dropna(axis=1, how='all')
    counts  = df['country_name'].value_counts().tolist()
    counts.insert(0,0)
    counts_cum = np.cumsum(counts)
    index_list = [[counts_cum[i-1],counts_cum[i]] for i in range(1,len(counts_cum))]
    with open("covid_19_1/within_grouping.txt", "wb") as fp:
        pickle.dump(index_list, fp)
    dataset_colums =df.columns.tolist()[2:]

    imputer = KNNImputer(n_neighbors=1, weights="uniform")
    vals  = imputer.fit_transform(df.values[:,2:])
    df = pd.DataFrame(vals,columns=dataset_colums)
    df = generate_autocorrelated_treatment(df)


    # df = df.drop(['npi_healthcare_investment','npi_vaccine_investment','npi_fiscal_measures','npi_international_support'],axis=1)
    x = [ ]
    z = df.columns.tolist()[28:]
    for el in df.columns.tolist():
        if 'npi' in el:
            x.append(el)
    for el in ['npi_healthcare_investment','npi_vaccine_investment','npi_fiscal_measures','npi_international_support']:
        x.remove(el)
    x.append('auto_corr_ref')
    z.append('cases_total_per_million')
    z = z+['npi_healthcare_investment','npi_vaccine_investment','npi_fiscal_measures','npi_international_support'] #lag cases to previous variables
    y = ['cases_new_per_million']
    print(x)
    print(y)
    print(z)
    Y_pd = df[y]
    Z_cont = df[z]
    Y_pd.to_csv(f"./covid_19_1/covid_Y.csv", index=False)
    Z_cont.to_csv(f"./covid_19_1/covid_Z_cont.csv", index=False)
    X_mult = df[x]
    Z_mult = Z_cont
    print(Z_cont.columns.tolist())
    # save_torch_mult(X_mult, Y_pd, Z_mult, './covid_19_1/', f'data_treatment_mult.pt')
    df = generate_autocorrelated_treatment(df)
    treatment_indices = [0,1,2,3,4,-2,-1]
    for i in treatment_indices:
        x_treat = [x[i]]
        z_cat=[]
        for el in x:
            if el == 'auto_corr_ref':
                continue
            elif el!=x[i]:
                z_cat.append(el)

        X_pd = df[x_treat]
        Z_cat = df[z_cat]
        X_pd.to_csv(f"./covid_19_1/covid_T={x[i]}.csv",index = False)
        Z_cat.to_csv(f"./covid_19_1/covid_Z_cat={x[i]}.csv",index = False)
        save_torch(X_pd,Y_pd,Z_cat,Z_cont,'./covid_19_1/',f'data_treatment={x[i]}.pt')
