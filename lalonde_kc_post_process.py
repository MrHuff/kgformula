from kchsic_categorical_power import *
if __name__ == '__main__':

    # dataset_names = ['twins']
    # for m in  [1000,2500,]:
    #     for dname in dataset_names:
    #         # bench = pd.read_csv(f"{dname}_bench_1d_{m}.csv").values.squeeze()
    #         # power = round(calc_power(bench, 0.05), 3)
    #         # plt.hist(bench, bins=[i / 25 for i in range(0, 26)])
    #         # plt.xlabel('p-values')
    #         # plt.ylabel('Frequency')
    #         # plt.title(rf'Power($\alpha=0.05$) = {power}')
    #         # plt.savefig(f'{dname}_pvals_bench_{m}.jpg', bbox_inches='tight',
    #         #             pad_inches=0.05)
    #         # plt.clf()
    #         for est in ['NCE_Q','real_TRE_Q']:
    #             for sep in [True]:
    #                 res = torch.load(f'twins_pvals_{est}_{sep}_{m}_2_linear.pt')
    #                 pvals = res['pvals'].numpy()
    #                 power = round(calc_power(pvals,0.05),3)
    #
    #                 plt.hist(pvals, bins=[i / 25 for i in range(0, 26)])
    #                 plt.xlabel('p-values')
    #                 plt.ylabel('Frequency')
    #                 plt.title(rf'Power($\alpha=0.05$) = {power}')
    #                 plt.savefig(f'{dname}_pvals_{est}_{sep}_{m}_linear.jpg', bbox_inches='tight',
    #                             pad_inches=0.05)
    #                 plt.clf()
    treatments = ['npi_school_closing', 'npi_workplace_closing', 'npi_cancel_public_events',
                  'npi_gatherings_restrictions', 'npi_close_public_transport', 'npi_stay_at_home',
                  'npi_internal_movement_restrictions', 'npi_international_travel_controls', 'npi_income_support',
                  'npi_debt_relief', 'npi_fiscal_measures', 'npi_international_support', 'npi_public_information',
                  'npi_testing_policy', 'npi_contact_tracing', 'npi_masks', 'auto_corr_ref']
    treatment_indices = [0, 1, 2, 3, 4, -1,-2]
    # treatment_indices = [-1]
    fold_res = 'weekly_covid_within_n_blocks_res'
    for m in [2500]:
        for n_blocks in [2, 3, 4, 5]:
            for ts in [True]:
                for within_grouping in [True]:
                    for treat_idx in treatment_indices:
                        treat = treatments[treat_idx]
                        for est in ['NCE_Q','real_TRE_Q']:
                            for sep in [False]:
                                res = torch.load(f'{fold_res}/covid_pvals_ts={ts}_{within_grouping}_{est}_{sep}_{m}_{treat}_nblocks={n_blocks}_.pt')
                                pvals = res['pvals'].numpy()
                                power = round(calc_power(pvals,0.05),3)
                                plt.hist(pvals, bins=[i / 25 for i in range(0, 26)])
                                plt.xlabel('p-values')
                                plt.ylabel('Frequency')
                                plt.title(fr'Power($\alpha=0.05$) = {power}')
                                plt.savefig(f'{fold_res}/covid_pvals_ts={ts}_{within_grouping}_{est}_{sep}_{m}_{treat}_nblocks={n_blocks}.jpg', bbox_inches='tight',
                                            pad_inches=0.05)
                                plt.clf()

    # dataset_names = ['lalonde']
    # m=100
    # for dname in dataset_names:
    #     bench = pd.read_csv(f"{dname}_bench_1d.csv").values.squeeze()
    #     power = round(calc_power(bench, 0.05), 3)
    #     plt.hist(bench, bins=[i / 25 for i in range(0, 26)])
    #     plt.xlabel('p-values')
    #     plt.ylabel('Frequency')
    #     plt.title(rf'Power($\alpha=0.05$) = {power}')
    #     plt.savefig(f'{dname}_pvals_bench_{m}.jpg', bbox_inches='tight',
    #                 pad_inches=0.05)
    #     plt.clf()
    #     for est in ['NCE_Q','real_TRE_Q']:
    #         for sep in [True]:
    #             res = torch.load(f'{dname}_pvals_{est}_{sep}_linear.pt')
    #             pvals = res['pvals'].numpy()
    #             power = round(calc_power(pvals,0.05),3)
    #
    #             plt.hist(pvals, bins=[i / 25 for i in range(0, 26)])
    #             plt.xlabel('p-values')
    #             plt.ylabel('Frequency')
    #             plt.title(rf'Power($\alpha=0.05$) = {power}')
    #             plt.savefig(f'{dname}_pvals_{est}_{sep}_linear.jpg', bbox_inches='tight',
    #                         pad_inches=0.05)
    #             plt.clf()


