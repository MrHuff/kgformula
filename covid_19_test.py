import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
import os
pd.options.display.max_rows = 10000
pd.options.display.max_columns = 1000
categorical = [
        'School closing',
        'Workplace closing',
       'Cancel public events',
        'Restrictions on gatherings',
       'Close public transport',
        'Stay at home requirements',
       'Restrictions on internal movement',
        'International travel controls',
       'Income support',
        'Debt/contract relief',
        'Testing policy',
        'Contact tracing',
        'Masks',
        'Public information campaigns',
        'ISO'
    ]
float = ['Emergency investment in healthcare',
             'new_cases',
             'Population Density',
             'Population',
             'aged_70_older',
             'median_age_y',
             'total_cases',
             'International support',
             'hospital_beds_per_100k',
             'total_deaths_per_million',
             'hospital_beds (per 1000)',
             'new_deaths',
             'female_smokers',
             'Population Male',
             'extreme_poverty',
             'population_density',
             'Population Female',
             'new_deaths_per_million',
             'total_cases_per_million',
             'nurses (per 1000)',
             'new_tests',
             'total_tests_per_thousand',
             'days_since_first_death',
             'population',
             'total_tests',
             'median_age_x',
             'aged_65_older',
             'Smoking prevalence, total, ages 15+',
             'Population in Urban Agglomerations',
             'diabetes_prevalence',
             'male_smokers',
             'days_since_first_case',
             'total_deaths',
             'cvd_death_rate',
             'Investment in vaccines',
             'new_cases_per_million',
             'Fiscal measures',
             'Population of Compulsory School Age',
             'physicians (per 1000)',
             'Mortality rate, adult, female (per 1,000 female adults)',
             'new_tests_per_thousand',
             'Mortality rate, adult, male (per 1,000 male adults)',
             'handwashing_facilities',
             'gdp_per_capita'
         ]

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
    list = df['CountryName'].unique().tolist()
    ret_list = []
    for el in list:
        subset = df[df['CountryName']==el]
        if not subset.isnull().values.any():
            ret_list.append(el)
    return ret_list

def transform(df):
    col_names = df.columns.tolist()
    _cat = []
    for el in col_names:
        if el in categorical:
            _cat.append(el)
    if _cat:
        cat_data = cat_fix(df[_cat])
        df = df.drop(columns=_cat)
        if df.shape[1]>0:
            df = normalize(df)
        df = pd.concat([df, cat_data], axis=1)
    else:
        if df.shape[1]>0:
            df = normalize(df)
    return df

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
    w={ 'indicator':[True]*Z_cat.shape[1]+Z_cont.shape[1]*[False],'index_lists':col_stats_list
}
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


if __name__ == '__main__':
    # df = pd.read_csv(
    #     'https://raw.githubusercontent.com/rs-delve/covid19_datasets/master/dataset/combined_dataset_latest.csv',
    #     parse_dates=['DATE'])
    #
    df = pd.read_csv("combined_dataset_latest.csv")
    df = df.drop(columns=['StringencyIndex',
                          'ISO',
                          'Retail and Recreation',
                          'Grocery and Pharmacy',
                          'Parks',
                          'Transit Stations',
                          'Workplaces',
                          'Residential'])
    cols_1 = df.columns.tolist()
    country_list = filter_nan_countries(df)
    cl_2 = df['CountryName'].unique().tolist()
    print(set(cl_2)-set(country_list))
    df = df[df['CountryName'].isin(country_list)]
    print(df.shape)
    print(df.columns)
    #new_deaths_per_million
    #
    y = ['new_cases_per_million','new_deaths_per_million'] #Need to consider repeating case... Confounder against testing policy. new_deaths_per_million
    x = [
        'School closing',
        'Workplace closing',
        'Cancel public events',
        'Restrictions on gatherings',
        'Close public transport',
        'Stay at home requirements',
        'Restrictions on internal movement',
        'International travel controls',
        'Testing policy',
        'Contact tracing',
        'Masks',
    ]

    z = [
        'International support',
        'Fiscal measures',
        'extreme_poverty',
        'physicians (per 1000)',
        'Mortality rate, adult, female (per 1,000 female adults)',
        'Mortality rate, adult, male (per 1,000 male adults)',
        'Population of Compulsory School Age',
        'hospital_beds (per 1000)',
        'nurses (per 1000)',
        'median_age_y',
        'Population Density',
        'aged_65_older',
        'gdp_per_capita',
        'new_tests_per_thousand'
    ]
    Y_pd = df[y]
    Z_cont = df[z]
    Y_pd.to_csv(f"./covid_19_1/covid_Y.csv", index=False)
    Z_cont.to_csv(f"./covid_19_1/covid_Z_cont.csv", index=False)

    X_mult = df[x]
    Z_mult = Z_cont
    Z_cat = []

    save_torch_mult(X_mult, Y_pd, Z_mult, './covid_19_1/', f'data_treatment_mult.pt')

    # treatment_indices = [0,1,2,3,4,-1]
    # for i in treatment_indices:
    #     x_treat = [x[i]]
    #     z_cat=[]
    #     for el in x:
    #         if el!=x[i]:
    #             z_cat.append(el)
    #
    #     X_pd = df[x_treat]
    #     Z_cat = df[z_cat]
    #     X_pd.to_csv(f"./covid_19_1/covid_T={x[i]}.csv",index = False)
    #     Z_cat.to_csv(f"./covid_19_1/covid_Z_cat={x[i]}.csv",index = False)
    #     # X_pd = transform(X_pd)
    #     # Y_pd = transform(Y_pd)
    #     # Z_pd = transform(Z_pd)
    #     save_torch(X_pd,Y_pd,Z_cat,Z_cont,'./covid_19_1/',f'data_treatment={x[i]}.pt')
