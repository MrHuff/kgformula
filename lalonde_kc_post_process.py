from kchsic_categorical_power import *
if __name__ == '__main__':

    bench = pd.read_csv("lalonde_bench_1d.csv").values.squeeze()
    power = round(calc_power(bench, 0.05), 3)
    plt.hist(bench, bins=[i / 25 for i in range(0, 26)])
    plt.xlabel('p-values')
    plt.ylabel('Frequency')
    plt.title(rf'Power($\alpha=0.05$) = {power}')
    plt.savefig(f'lalonde_pvals_bench.jpg', bbox_inches='tight',
                pad_inches=0.05)
    plt.clf()
    for est in ['NCE_Q']:
        for sep in [True]:
            res = torch.load(f'lalonde_pvals_{est}_{sep}.pt')
            pvals = res['pvals'].numpy()
            power = round(calc_power(pvals,0.05),3)

            plt.hist(pvals, bins=[i / 25 for i in range(0, 26)])
            plt.xlabel('p-values')
            plt.ylabel('Frequency')
            plt.title(rf'Power($\alpha=0.05$) = {power}')
            plt.savefig(f'lalonde_pvals_{est}_{sep}.jpg', bbox_inches='tight',
                        pad_inches=0.05)



