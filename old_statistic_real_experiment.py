import os
import torch
from kgformula.kernel_baseline_utils import *


job_res_name= 'old_statistic_real_world_data'

if not os.path.exists(job_res_name):
    os.makedirs(job_res_name)


for dat_name in ['twins','lalonde']:
    if dat_name=='twins':
        X, Y, Z, ind_dict = torch.load(f'{dat_name}.pt')
        m_list=[5000]
    else:
        df = pd.read_csv("lalonde.csv")
        X = torch.from_numpy(df['treat'].values).unsqueeze(-1).float()
        Y = torch.from_numpy(df['re78'].values).unsqueeze(-1).float()
        Z_df = df[['age', 'education', 'black', 'hispanic', 'married', 'nodegree', 're74', 're75']]
        Z = torch.from_numpy(Z_df.values).float()
        m_list =[100,150,200]
    n=X.shape[0]
    for m in m_list:
        for use_dummy_y in [True,False]:
            for var in [1]:
                for est in ['real_weights']:
                    args = {'qdist':2,
                            'job_character': {},
                            'bootstrap_runs':250,
                            'job_type':'old_statistic',
                            'est_params': {'lr': 1e-3, #use really small LR for TRE. Ok what the fuck is going on...
                                                                       'max_its': 100,

                                                                       'width': 32,
                                                                       'layers':2,
                                                                       'mixed': False,
                                                                       'bs_ratio': 1e-1,
                                                                       'val_rate': 0.1,
                                                                       'n_sample': 250,
                                                                       'criteria_limit': 0.05,
                                                                       'kill_counter': 2,
                                           },
                            'estimator': est,

                            'cuda': True,
                            'device': 'cuda:0',
                            'n':10000,
                            'unique_job_idx':9999,
                            'variant': var
                            }
                    pval_list = []
                    for i in range(100):
                        # i=21
                        print('seed: ',i)
                        torch.random.manual_seed(i)
                        perm = torch.randperm(n)[:m]
                        x = X[perm,:]
                        if use_dummy_y:
                            y = torch.randn_like(x)
                        else:
                            y = Y[perm]
                        z = Z[perm]
                        try:
                            sim_obj =  kernel_baseline(args) #Why getting nans?
                            pval, ref = sim_obj.run_data(x,y,z)
                        except Exception as e:
                            print (e)
                        pval_list.append(pval)
                    pvals = torch.tensor(pval_list)
                    torch.save({'pvals':pvals,'config':args},f'{job_res_name}/{dat_name}_{use_dummy_y}_{m}_pvals.pt')



