from matplotlib import pyplot as plt
from sklearn.decomposition import PCA,KernelPCA
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
import torch
from create_plots import *
def get_biggest_bin(dat):
    vals, counts = np.unique(dat, return_counts=True)
    print(vals,counts)
    max_ind = np.argmax(counts)
    return vals[max_ind],vals.tolist()
if __name__ == '__main__':
    s = StandardScaler()
    for treatment in ['Masks','Cancel public events']:
        X, Y, Z, ind_dict = torch.load(f'covid_19_1/data_treatment={treatment}.pt')
        plt.scatter(X.numpy(),Y.numpy())
        plt.xlabel(treatment)
        plt.ylabel('New cases per million')
        plt.savefig(f'covid_scatter_{treatment}_pre.png', bbox_inches='tight',
                    pad_inches=0.05)
        plt.clf()
        Z = s.fit_transform(Z.numpy())
        pca = PCA(n_components=2)
        z_pca = pca.fit_transform(Z)
        explained_variance = np.var(z_pca, axis=0)
        explained_variance_ratio = explained_variance / np.sum(explained_variance)

        est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
        disc = est.fit_transform(z_pca)
        pca_dim = 0
        biggest_first_dim,uniques = get_biggest_bin(disc[:, pca_dim])
        # m1 = disc[:, 0] == biggest_first_dim
        # X = X[m1]
        # Y= Y[m1]
        # disc = disc[m1]
        # biggest_second_dim = get_biggest_bin(disc[:, 1])
        # m2 = disc[:, 1] == biggest_second_dim
        # disc = disc[m2]
        # X = X[m2].numpy()
        # Y = Y[m2].numpy()
        symbols = ['.','*','o']
        for i in uniques:
            x = X[disc[:,pca_dim]==i].numpy()
            y = Y[disc[:,pca_dim]==i].numpy()
            plt.scatter(x,y,label=i,alpha=0.25,marker=symbols[int(i)])
        plt.legend()
        plt.xlabel(treatment)
        plt.ylabel('New cases per million')
        plt.savefig(f'covid_scatter_{treatment}_{pca_dim}.png', bbox_inches='tight',
                    pad_inches=0.05)
        plt.clf()









