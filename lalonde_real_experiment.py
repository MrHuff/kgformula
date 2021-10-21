import pandas as pd
import torch
from kgformula.utils import simulation_object_rule_new
df = pd.read_csv("lalonde.csv")

X = torch.from_numpy(df['treat'].values).unsqueeze(-1).float()
Y = torch.from_numpy(df['re78'].values).unsqueeze(-1).float()
Z = torch.from_numpy(df[['age','education','black','hispanic','married','nodegree','re74','re75']].values).float()

n = X.shape[0]
m = 100

for est in ['TRE_Q','real_TRE_Q']:
    for sep in [True]:
        args = {'qdist':2,
                'bootstrap_runs':250,
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
                                                            'kappa':10 if est=='NCE_Q' else 1,
                                                           'm': 4,
                                                           'separate': sep
                               },
                'estimator': est,
                'cuda': True,
                'device': 'cuda:0',
                'n':100,
                'unique_job_idx':9998,
                'variant': 1
                }
        pval_list = []
        for i in range(100):
            # i=21
            print('seed: ',i)
            torch.random.manual_seed(i)
            perm = torch.randperm(n)[:m]
            x = X[perm]
            y = Y[perm]
            z = Z[perm]
            # try:
            sim_obj = simulation_object_rule_new(args=args) #Why getting nans?
            pval, ref = sim_obj.run_data(x,y,z)
            # except Exception as e:
            #     print (e)
            pval_list.append(pval)
        pvals = torch.tensor(pval_list)
        torch.save({'pvals':pvals,'config':args},f'lalonde_pvals_{est}_{sep}_rbf.pt')




