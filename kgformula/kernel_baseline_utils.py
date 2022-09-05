import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegressionCV
import torch
import numpy as np
from pycox.preprocessing.feature_transforms import *
from sklearn.model_selection import train_test_split
from kgformula.cfme_baseline import *
from kgformula.old_statistic import *
import os
import tqdm
import pickle
from scipy.stats import kstest

def categorical_transformer(X,cat_cols,cont_cols):
    c = OrderedCategoricalLong()
    for el in cat_cols:
        X[:,el] = c.fit_transform(X[:,el])
    cat_cols = cat_cols
    if cat_cols:
        unique_cat_cols = X[:,cat_cols].max(axis=0).tolist()
        unique_cat_cols = [el + 1 for el in unique_cat_cols]
    else:
        unique_cat_cols = []
    X_cont=X[cont_cols]
    X_cat=X[cat_cols]
    return X_cont,X_cat,unique_cat_cols
class sklearn_propensity_estimator():
    def __init__(self, X_tr, T_tr, X_val, T_val, nn_params, bs=100, epochs=100, device='cuda:0', X_cat_tr=[],
                 X_cat_val=[]):
        self.X  = X_tr.cpu().numpy()
        self.T  = T_tr.squeeze().cpu().numpy()
        self.clf = LogisticRegressionCV(cv=10, random_state=0,verbose=False)
        self.X_val = X_val.cpu().numpy()
        self.T_val = T_val.squeeze().cpu().numpy()
        self.device = device
    def fit(self,tmp=None):
        self.clf.fit(self.X,self.T)
        self.best = self.clf.score(self.X_val,self.T_val)

    def predict(self,X,T=[],x_cat=[]):
        return torch.from_numpy(self.clf.predict_proba(X.cpu().numpy())[:,-1]).float().to(self.device).unsqueeze(-1)

class baseline_test_class_incorrect():
    def __init__(self,X,T,Y,W,nn_params,training_params,cat_cols=[]):
        # split data
        self.cat_cols = cat_cols
        self.unique_cat_cols = []
        if cat_cols:
            self.cont_cols = list(set([i for i in range(X.shape[1])]) - set(cat_cols))
            self.X_cont, self.X_cat, self.unique_cat_cols = categorical_transformer(X, self.cat_cols,
                                                                                    cont_cols=self.cont_cols)
        self.nn_params = nn_params
        self.nn_params['cat_size_list'] = self.unique_cat_cols
        self.nn_params['d_in_x'] = self.X_cont.shape[1] if cat_cols else X.shape[1]
        indices = np.arange(X.shape[0])
        tr_ind, tst_ind, tmp_T, self.tst_T = train_test_split(indices, T, test_size=0.5)
        tr_ind_2, val_ind, self.tr_T, self.val_T = train_test_split(tr_ind, tmp_T, test_size=0.1)
        self.training_params = training_params
        self.split_into_cont_cat(X, Y, W, tr_ind_2, val_ind, tst_ind)

    def split_into_cont_cat(self, X, Y, W, tr_ind_2, val_ind, tst_ind):
        kw = ['tr', 'val', 'tst']
        indices = [tr_ind_2, val_ind, tst_ind]
        for w, idx in zip(kw, indices):
            setattr(self, f'{w}_idx', idx)

            setattr(self, f'{w}_X', X[idx])
            setattr(self, f'{w}_Y', Y[idx])
            setattr(self, f'{w}_W', W[idx])
            if self.cat_cols:
                setattr(self, f'{w}_X_cont', self.X_cont[idx])
                setattr(self, f'{w}_X_cat', self.X_cat[idx])
            else:
                setattr(self, f'{w}_X_cat', [])
                setattr(self, f'{w}_X_cont', X[idx])
    @staticmethod
    def calculate_pval_symmetric(bootstrapped_list, test_statistic):
        pval_right = 1 - 1 / (bootstrapped_list.shape[0] + 1) * (1 + (bootstrapped_list <= test_statistic).sum())
        pval_left = 1 - pval_right
        pval = 2 * min([pval_left.item(), pval_right.item()])
        return pval

    def run_test(self,seed):
        #train classifier

        if self.training_params['oracle_weights']:
            self.e = torch.from_numpy(self.tst_W)
        else:
            self.classifier = sklearn_propensity_estimator(self.tr_X_cont, self.tr_T, self.val_X_cont,
                                                   self.val_T, nn_params=self.nn_params,
                                                   bs=self.training_params['bs'], X_cat_val=self.val_X_cat,
                                                   X_cat_tr=self.tr_X_cat, epochs=self.training_params['epochs'])
            self.classifier.fit(self.training_params['patience'])
            print('classifier val auc: ', self.classifier.best)
            self.e = self.classifier.predict(self.tst_X,self.tst_T,self.tst_X_cat)
            self.e = torch.nan_to_num(self.e, nan=0.5, posinf=0.5)

        self.test=baseline_test_gpu_incorrect_og(self.tst_Y,e=self.e,T=self.tst_T,permutations=self.training_params['permutations'])
        perm_stats,self.tst_stat = self.test.permutation_test()
        self.perm_stats = perm_stats
        self.pval = self.calculate_pval_symmetric(self.perm_stats,self.tst_stat )
        output = [seed,self.pval,self.tst_stat]
        return output#+perm_stats.tolist()
class baseline_test_old(baseline_test_class_incorrect):

    #Introduce both correct and incorrect permutation for reference!
    def __init__(self,X,T,Y,W,nn_params,training_params,cat_cols=[]):
        super(baseline_test_old, self).__init__(X,T,Y,W,nn_params,training_params,cat_cols)

    def run_test(self, seed):
        self.test = old_statistic(self.tr_T,self.tr_Y,self.tr_X,self.tst_T,self.tst_Y,self.tst_X ,permutations =self.training_params['permutations'] ,device='cuda:0')
        perm_stats, self.tst_stat = self.test.permutation_test()
        self.perm_stats = perm_stats
        self.pval = self.calculate_pval_symmetric(self.perm_stats, self.tst_stat)
        output = [seed, self.pval, self.tst_stat]
        return output #+ perm_stats.tolist()

class kernel_baseline():
    def __init__(self,args):
        self.args=args
        self.cuda = self.args['cuda']
        self.device = self.args['device']
        self.validation_chunks = 10
        self.validation_over_samp = 10
        self.variant = self.args['variant']
        self.max_validation_samples =  self.args['n']//4
        self.train_params={}
    @staticmethod
    def get_level(level, p_values):
        total_pvals = len(p_values)
        power = sum(p_values <= level) / total_pvals
        return power
    # def calc_summary_stats(self,pvals):
    #     output=[]
    #     ks_stat, ks_pval = kstest(pvals, 'uniform')
    #     levels = [0.01, 0.05, 0.1]
    #     for l in levels:
    #         output.append(self.get_level(l,pvals))
    #     output.append(ks_pval)
    #     output.append(ks_stat)
    #     return output
    def run(self):
        job_dir = self.args['job_dir']
        data_dir = self.args['data_dir']
        seeds_a = self.args['seeds_a']
        seeds_b = self.args['seeds_b']
        self.job_character = self.args['job_character']
        self.qdist = self.args['qdist']
        self.bootstrap_runs = self.args['bootstrap_runs']
        self.train_params['oracle_weights']=False
        self.train_params['epochs']=100
        self.train_params['bs']=self.args['n']//4
        self.train_params['patience']=10
        self.train_params['permutations']=self.args['bootstrap_runs']
        estimator = self.args['job_type']
        if not os.path.exists(f'./{job_dir}_results'):
            os.makedirs(f'./{job_dir}_results')
        data_col = []
        c=0
        # summary_job_cols = ['pow_001','pow_005','pow_010','KS-pval','KS-stat']
        # columns = ['seed','pval','tst_stat']+[f'perm_{i}' for i in range(self.train_params['permutations'])]
        for i in tqdm.trange(seeds_a, seeds_b):
            X, Y, Z, _w = torch.load(f'./{data_dir}/data_seed={i}.pt')
            X, Y, Z, _w = X.cuda(self.device), Y.cuda(self.device), Z.cuda(self.device), _w.cuda(self.device)
            if estimator == 'cfme':
                tst=baseline_test_class_incorrect(Z,X,Y,_w,nn_params={},training_params=self.train_params)
            elif estimator =='old_statistic':
                tst=baseline_test_old(Z,X,Y,_w,nn_params={},training_params=self.train_params)
            try:
                out = tst.run_test(i)
                data_col.append(out)
            except Exception as e:
                print(e)
                c += 1
            if c > 10:
                raise Exception('Dude something is seriously wrong with your data or the method please debug')
        dat=np.array(data_col)
        ks_data=[]
        p_value_array = torch.tensor(dat[:,1])
        ref_metric_array = torch.tensor(dat[:,2])
        q_fac_array = torch.tensor([1.0]*100)
        ks_stat, p_val_ks_test = kstest(p_value_array.numpy(), 'uniform')
        print(f'KS test Uniform distribution test statistic: {ks_stat}, p-value: {p_val_ks_test}')
        ks_data.append([ks_stat, p_val_ks_test])
        df = pd.DataFrame(ks_data, columns=['ks_stat', 'p_val_ks_test'])
        s = df.describe()
        results_dict = {
            'p_value_array':p_value_array,
            'ref_metric_array':ref_metric_array,
            'q_fac_array':q_fac_array,
            'df':df,
            's':s,
        }
        unique_job_idx=self.args['unique_job_idx']
        with open(f'./{job_dir}_results/results_{unique_job_idx}.pickle', 'wb') as handle:
            pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def run_data(self,X,Y,Z):
        self.job_character = self.args['job_character']
        self.qdist = self.args['qdist']
        self.bootstrap_runs = self.args['bootstrap_runs']
        self.train_params['oracle_weights'] = False
        self.train_params['epochs'] = 100
        self.train_params['bs'] = self.args['n'] // 4
        self.train_params['patience'] = 10
        self.train_params['permutations'] = self.args['bootstrap_runs']
        estimator = self.args['job_type']
        X, Y, Z = X.cuda(self.device), Y.cuda(self.device), Z.cuda(self.device)
        _w = torch.ones(X.shape[0]).cuda(self.device)
        if estimator == 'cfme':
            tst = baseline_test_class_incorrect(Z, X, Y, _w, nn_params={}, training_params=self.train_params)
        elif estimator == 'old_statistic':
            tst = baseline_test_old(Z, X, Y, _w, nn_params={}, training_params=self.train_params)
        s,pval,ref = tst.run_test(0)
        return pval,ref

