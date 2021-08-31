from kchsic_categorical_power import *
if __name__ == '__main__':
    dataset_names = ['twins']
    for m in [1000,2500,5000]:
        for dname in dataset_names:
            bench = pd.read_csv(f"{dname}_bench_1d_{m}.csv").values.squeeze()
            power = round(calc_power(bench, 0.05), 3)
            plt.hist(bench, bins=[i / 25 for i in range(0, 26)])
            plt.xlabel('p-values')
            plt.ylabel('Frequency')
            plt.title(rf'Power($\alpha=0.05$) = {power}')
            plt.savefig(f'{dname}_pvals_bench_{m}.jpg', bbox_inches='tight',
                        pad_inches=0.05)
            plt.clf()
            for est in ['NCE_Q','real_TRE_Q']:
                for sep in [True]:
                    res = torch.load(f'{dname}_pvals_{est}_{sep}_{m}.pt')
                    pvals = res['pvals'].numpy()
                    power = round(calc_power(pvals,0.05),3)

                    plt.hist(pvals, bins=[i / 25 for i in range(0, 26)])
                    plt.xlabel('p-values')
                    plt.ylabel('Frequency')
                    plt.title(rf'Power($\alpha=0.05$) = {power}')
                    plt.savefig(f'{dname}_pvals_{est}_{sep}_{m}.jpg', bbox_inches='tight',
                                pad_inches=0.05)
                    plt.clf()

    dataset_names = ['lalonde']
    for dname in dataset_names:
        bench = pd.read_csv(f"{dname}_bench_1d.csv").values.squeeze()
        power = round(calc_power(bench, 0.05), 3)
        plt.hist(bench, bins=[i / 25 for i in range(0, 26)])
        plt.xlabel('p-values')
        plt.ylabel('Frequency')
        plt.title(rf'Power($\alpha=0.05$) = {power}')
        plt.savefig(f'{dname}_pvals_bench_{m}.jpg', bbox_inches='tight',
                    pad_inches=0.05)
        plt.clf()
        for est in ['NCE_Q','real_TRE_Q']:
            for sep in [True,False]:
                res = torch.load(f'{dname}_pvals_{est}_{sep}.pt')
                pvals = res['pvals'].numpy()
                power = round(calc_power(pvals,0.05),3)

                plt.hist(pvals, bins=[i / 25 for i in range(0, 26)])
                plt.xlabel('p-values')
                plt.ylabel('Frequency')
                plt.title(rf'Power($\alpha=0.05$) = {power}')
                plt.savefig(f'{dname}_pvals_{est}_{sep}.jpg', bbox_inches='tight',
                            pad_inches=0.05)
                plt.clf()


