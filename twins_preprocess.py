import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn import preprocessing

types = {'adequacy': 'cat',
 'alcohol': 'bin',
 'anemia': 'bin',
 'birattnd': 'cat',
 'birmon': 'cyc',
 'bord': 'bin',
 'brstate': 'cat',
 'brstate_reg': 'cat',
 'cardiac': 'bin',
 'chyper': 'bin',
 'cigar6': 'cat',
 'crace': 'cat',
 'csex': 'bin',
 'data_year': 'cat',
 'dfageq': 'cat',
 'diabetes': 'bin',
 'dlivord_min': 'ord',
 'dmar': 'bin',
 'drink5': 'cat',
 'dtotord_min': 'ord',
 'eclamp': 'bin',
 'feduc6': 'cat',
 'frace': 'cat',
 'gestat10': 'cat',
 'hemo': 'bin',
 'herpes': 'bin',
 'hydra': 'bin',
 'incervix': 'bin',
 'infant_id': 'index do not use',
 'lung': 'bin',
 'mager8': 'cat',
 'meduc6': 'cat',
 'mplbir': 'cat',
 'mpre5': 'cat',
 'mrace': 'cat',
 'nprevistq': 'cat',
 'orfath': 'cat',
 'ormoth': 'cat',
 'othermr': 'bin',
 'phyper': 'bin',
 'pldel': 'cat',
 'pre4000': 'bin',
 'preterm': 'bin',
 'renal': 'bin',
 'rh': 'bin',
 'stoccfipb': 'cat',
 'stoccfipb_reg': 'cat',
 'tobacco': 'bin',
 'uterine': 'bin'}



X = pd.read_csv("TWINS/twin_pairs_T_3years_samesex.csv",index_col=[0])
Y = pd.read_csv("TWINS/twin_pairs_Y_3years_samesex.csv",index_col=[0])
Z = pd.read_csv("TWINS/twin_pairs_X_3years_samesex.csv",index_col=[0])

Z = Z.drop(['infant_id_1','Unnamed: 0.1','infant_id_0'],axis=1)
Z = Z.dropna()
rows_to_keep = Z.index.tolist()

X = X.iloc[rows_to_keep,:]
X['diff'] = X['dbirwt_1']-X['dbirwt_0']
Y = Y.iloc[rows_to_keep,:]
Y = Y['mort_0']-Y['mort_1']


if __name__ == '__main__':
    cat_cols = []
    for i,j in types.items():
        if j =='cat':
            cat_cols.append(i)
    col_counts = []
    col_stats_list = []
    col_index_list = [False]*Z.shape[1]
    for cat_col in cat_cols:
        col_index_list[Z.columns.get_loc(cat_col)]=True
    cat_cols = Z.iloc[:,col_index_list]

    for i in range(cat_cols.shape[1]):
        col_stats = cat_cols.iloc[:,i].unique().tolist()
        col_stats_list.append(col_stats)
        col_counts.append(len(col_stats))


    plt.hist(X.values)
    plt.show()
    plt.clf()
    X = (X-X.mean(0))/X.std(0)
    plt.hist(X.values)
    plt.show()

    torch.save((torch.from_numpy(X.values).float(), torch.from_numpy(Y.values).float().unsqueeze(-1), torch.from_numpy(Z.values).float(),{'indicator':col_index_list,'index_lists':col_stats_list}), 'twins.pt')
    print(cat_cols)
    print(Z.values[:,col_index_list])
    X.to_csv("twins_T.csv",index = False)
    cat_cols.to_csv("twins_z_cat.csv",index = False)
    cont_cols = Z.iloc[:,~np.array(col_index_list)]
    cont_cols.to_csv("twins_z_cont.csv",index = False)
    Y.to_csv("twins_Y.csv",index = False)







