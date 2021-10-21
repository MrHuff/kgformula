from kchsic_categorical_power import *
if __name__ == '__main__':

    dataset_names = ['twins']
    for m in  [1000,2500,]:
        for dname in dataset_names:
            # bench = pd.read_csv(f"{dname}_bench_1d_{m}.csv").values.squeeze()
            # power = round(calc_power(bench, 0.05), 3)
            # plt.hist(bench, bins=[i / 25 for i in range(0, 26)])
            # plt.xlabel('p-values')
            # plt.ylabel('Frequency')
            # plt.title(rf'Power($\alpha=0.05$) = {power}')
            # plt.savefig(f'{dname}_pvals_bench_{m}.jpg', bbox_inches='tight',
            #             pad_inches=0.05)
            # plt.clf()
            for est in ['NCE_Q','real_TRE_Q']:
                for sep in [True]:
                    res = torch.load(f'twins_pvals_{est}_{sep}_{m}_2_linear.pt')
                    pvals = res['pvals'].numpy()
                    power = round(calc_power(pvals,0.05),3)

                    plt.hist(pvals, bins=[i / 25 for i in range(0, 26)])
                    plt.xlabel('p-values')
                    plt.ylabel('Frequency')
                    plt.title(rf'Power($\alpha=0.05$) = {power}')
                    plt.savefig(f'{dname}_pvals_{est}_{sep}_{m}_linear.jpg', bbox_inches='tight',
                                pad_inches=0.05)
                    plt.clf()
    treatment_indices = [0, 1, 2, 3, 4, -2,-1]
    treatments = [
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
    'Masks','mult' ]
    dataset_names = ['covid']
    for m in [2500]:
        for dname in dataset_names:
            # bench = pd.read_csv(f"{dname}_bench_1d_{m}.csv").values.squeeze()
            # power = round(calc_power(bench, 0.05), 3)
            # plt.hist(bench, bins=[i / 25 for i in range(0, 26)])
            # plt.xlabel('p-values')
            # plt.ylabel('Frequency')
            # plt.title(rf'Power($\alpha=0.05$) = {power}')
            # plt.savefig(f'{dname}_pvals_bench_{m}.jpg', bbox_inches='tight',
            #             pad_inches=0.05)
            # plt.clf()
            for treat_idx in treatment_indices:
                treat = treatments[treat_idx]
                for est in ['NCE_Q']:
                    for sep in [True]:
                        res = torch.load(f'{dname}_pvals_{est}_{sep}_{m}_{treat}_.pt')
                        pvals = res['pvals'].numpy()
                        power = round(calc_power(pvals,0.05),3)
                        plt.hist(pvals, bins=[i / 25 for i in range(0, 26)])
                        plt.xlabel('p-values')
                        plt.ylabel('Frequency')
                        plt.title(rf'Power($\alpha=0.05$) = {power}')
                        plt.savefig(f'{dname}_pvals_{est}_{sep}_{m}_{treat}.jpg', bbox_inches='tight',
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


