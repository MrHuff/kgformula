import pandas as pd
from generate_covid_real_experiment import treatments,treatment_indices
from matplotlib import pyplot as plt
from bdhsic_categorical_power import *

if __name__ == '__main__':
    for n in [5000]:
        for t in treatment_indices:
            treat = treatments[t]
            pval_plot_data = pd.read_csv(f"covid_bench_T={treat}_n={n}.csv",index_col=0)
            pvals = pval_plot_data.values.squeeze()
            power = round(calc_power(pvals, 0.05), 3)
            plt.hist(pvals, bins=[i / 25 for i in range(0, 26)])
            plt.xlabel('p-values')
            plt.ylabel('Frequency')
            plt.title(rf'Power($\alpha=0.05$) = {power}')
            plt.savefig(f'covid_hdm_{treat}.jpg', bbox_inches='tight',
                        pad_inches=0.05)
            plt.clf()

    for n in [2500,5000]:
        for t in treatment_indices:
            treat = treatments[t]
            pval_plot_data = pd.read_csv(f"covid_bench_T={treat}_n={n}_dummy.csv",index_col=0)
            pvals = pval_plot_data.values.squeeze()
            power = round(calc_power(pvals, 0.05), 3)
            plt.hist(pvals, bins=[i / 25 for i in range(0, 26)])
            plt.xlabel('p-values')
            plt.ylabel('Frequency')
            plt.title(rf'Power($\alpha=0.05$) = {power}')
            plt.savefig(f'covid_hdm_{treat}_n={n}_dummy.jpg', bbox_inches='tight',
                        pad_inches=0.05)
            plt.clf()
