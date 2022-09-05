from bdhsic_categorical_power import *
if __name__ == '__main__':

    dataset_names = ['twins']
    job_res_name = 'twins_exp_perm'

    for m in  [5000]:
        bench = pd.read_csv(f"twins_bench_1d_5000_dummy.csv").values.squeeze()
        power = round(calc_power(bench, 0.05), 3)
        plt.hist(bench, bins=[i / 25 for i in range(0, 26)])
        plt.xlabel('p-values')
        plt.ylabel('Frequency')
        plt.title(rf'Power($\alpha=0.05$) = {power}')
        plt.savefig(f'{job_res_name}/twins_bench_1d_5000_dummy.jpg', bbox_inches='tight',
                    pad_inches=0.05)
        plt.clf()
        for use_dummy_y in [True, False]:
            for m in [5000]:
                for var in [1, 2]:
                    for est in ['NCE_Q','real_TRE_Q']:
                        for sep in [False]:
                            res = torch.load(f'{job_res_name}/twins_pvals_{est}_{sep}_{m}_{use_dummy_y}_kernel={var}.pt')
                            pvals = res['pvals'].numpy()
                            power = round(calc_power(pvals,0.05),3)

                            plt.hist(pvals, bins=[i / 25 for i in range(0, 26)])
                            plt.xlabel('p-values')
                            plt.ylabel('Frequency')
                            plt.title(rf'Power($\alpha=0.05$) = {power}')
                            plt.savefig(f'{job_res_name}/twins_pvals_{est}_{sep}_{m}_{use_dummy_y}_kernel={var}.jpg', bbox_inches='tight',
                                        pad_inches=0.05)
                            plt.clf()
    # treatments = ['npi_school_closing', 'npi_workplace_closing', 'npi_cancel_public_events',
    #               'npi_gatherings_restrictions', 'npi_close_public_transport', 'npi_stay_at_home',
    #               'npi_internal_movement_restrictions', 'npi_international_travel_controls', 'npi_income_support',
    #               'npi_debt_relief', 'npi_fiscal_measures', 'npi_international_support', 'npi_public_information',
    #               'npi_testing_policy', 'npi_contact_tracing', 'npi_masks', 'auto_corr_ref']
    # treatment_indices = [0, 1, 2, 3, 4, -1,-2]
    # # treatment_indices = [-1]
    # fold_res = 'weekly_covid_within_n_blocks_50_y_res'
    # for m in [2500]:
    #     for n_blocks in [2,3,  4]:
    #         for ts in [True]:
    #             for within_grouping in [True]:
    #                 for treat_idx in treatment_indices:
    #                     treat = treatments[treat_idx]
    #                     for est in ['NCE_Q','real_TRE_Q']:
    #                         for sep in [False]:
    #                             res = torch.load(f'{fold_res}/covid_pvals_ts={ts}_{within_grouping}_{est}_{sep}_{m}_{treat}_nblocks={n_blocks}_.pt')
    #                             pvals = res['pvals'].numpy()
    #                             power = round(calc_power(pvals,0.05),3)
    #                             plt.hist(pvals, bins=[i / 25 for i in range(0, 26)])
    #                             plt.xlabel('p-values')
    #                             plt.ylabel('Frequency')
    #                             plt.title(fr'Power($\alpha=0.05$) = {power}')
    #                             plt.savefig(f'{fold_res}/covid_pvals_ts={ts}_{within_grouping}_{est}_{sep}_{m}_{treat}_nblocks={n_blocks}.jpg', bbox_inches='tight',
    #                                         pad_inches=0.05)
    #                             plt.clf()

    job_res_name = 'lalonde_bdhsic_perm'
    bench = pd.read_csv(f"lalonde_bench_1d_dummy.csv").values.squeeze()
    power = round(calc_power(bench, 0.05), 3)
    plt.hist(bench, bins=[i / 25 for i in range(0, 26)])
    plt.xlabel('p-values')
    plt.ylabel('Frequency')
    plt.title(rf'Power($\alpha=0.05$) = {power}')
    plt.savefig(f'{job_res_name}/lalonde_pvals_bench_dummy.jpg', bbox_inches='tight',
                pad_inches=0.05)
    plt.clf()
    for est in ['NCE_Q','real_TRE_Q']:
        for m in [100, 150]:
            for var in [1, 2]:
                for sep in [False]:
                    for use_dummy_y in [True, False]:
                        try:
                            res = torch.load(f'{job_res_name}/lalonde_pvals_n={m}_{est}_{sep}_kernel={var}_dummy={use_dummy_y}.pt')
                            pvals = res['pvals'].numpy()
                            power = round(calc_power(pvals,0.05),3)

                            plt.hist(pvals, bins=[i / 25 for i in range(0, 26)])
                            plt.xlabel('p-values')
                            plt.ylabel('Frequency')
                            plt.title(rf'Power($\alpha=0.05$) = {power}')
                            plt.savefig(f'{job_res_name}/lalonde_pvals_n={m}_{est}_{sep}_kernel={var}_dummy={use_dummy_y}.jpg', bbox_inches='tight',
                                        pad_inches=0.05)
                            plt.clf()
                        except Exception as e:
                            pass

    """
    Process baselines
    """

    job_res_name = 'old_statistic_real_world_data'
    for dataset in ['twins','lalonde']:
        if dataset=='twins':
            m_list = [5000]
        else:
            m_list = [100,150,200]
        for m in m_list:
            for use_dummy_y in [True, False]:
                try:
                    res = torch.load(f'{job_res_name}/{dataset}_{use_dummy_y}_{m}_pvals.pt')
                    pvals = res['pvals'].numpy()
                    power = round(calc_power(pvals,0.05),3)

                    plt.hist(pvals, bins=[i / 25 for i in range(0, 26)])
                    plt.xlabel('p-values')
                    plt.ylabel('Frequency')
                    plt.title(rf'Power($\alpha=0.05$) = {power}')
                    plt.savefig(f'{job_res_name}/{dataset}_{use_dummy_y}_{m}.jpg', bbox_inches='tight',
                                pad_inches=0.05)
                    plt.clf()
                except Exception as e:
                    print(e)