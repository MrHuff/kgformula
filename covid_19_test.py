import pandas as pd
pd.options.display.max_rows = 10000
pd.options.display.max_columns = 1000

def cat_fix(df):
    pass

def normalize():

    pass

if __name__ == '__main__':
    df = pd.read_csv("combined_dataset_latest.csv")
    df = df.drop(columns=['StringencyIndex',
                          'ISO',
                          'Retail and Recreation',
                          'Grocery and Pharmacy',
                          'Parks',
                          'Transit Stations',
                          'Workplaces',
                          'Residential'])
    print(df['CountryName'].unique())
    print(df.shape)
    cols_1 = df.columns.tolist()
    df = df.dropna(axis='columns')
    print(df.shape)
    cols_2 = df.columns.tolist()
    set_minus = set(cols_1)-set(cols_2) #Are these covariates important?!
    print(set_minus)
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
        'Contact tracing'
        'Masks',
        'Public information campaigns',
        'ISO'
    ]
    y = []
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
             'Masks',
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
             'Contact tracing',
             'Fiscal measures',
             'Population of Compulsory School Age',
             'physicians (per 1000)',
             'Mortality rate, adult, female (per 1,000 female adults)',
             'new_tests_per_thousand',
             'Mortality rate, adult, male (per 1,000 male adults)',
             'handwashing_facilities',
             'gdp_per_capita']
    y_cat = []
    x = []
    x_cat = []
    z = []
    z_cat = []
    # subset = df[df['ISO']=='USA']
    # print(subset)
    # print(subset[float])
