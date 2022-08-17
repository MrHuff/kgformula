import pandas as pd
import torch
import os
from kgformula.utils import simulation_object_rule_new,simulation_object_rule_perm
# os.environ["CUDA_VISIBLE_DEVICES"]="0"


df = pd.read_csv("lalonde.csv")
X = torch.from_numpy(df['treat'].values).unsqueeze(-1).float()
Y = torch.from_numpy(df['re78'].values).unsqueeze(-1).float()
Z_df = df[['age','education','black','hispanic','married','nodegree','re74','re75']]
Z = torch.from_numpy(Z_df.values).float()
cat_cols = ['education','black','hispanic','married','nodegree']
col_stats_list=[]
col_counts=[]
col_index_list = [False] * Z.shape[1]

for cat_col in cat_cols:
    col_index_list[Z_df.columns.get_loc(cat_col)] = True
for cat_col in cat_cols:
    col_stats = Z_df[cat_col].unique().tolist()
    col_stats_list.append(col_stats)
    col_counts.append(len(col_stats))
print(col_index_list)
print(Z.shape)
cat_data = {'indicator':col_index_list,'index_lists':col_stats_list}
n = X.shape[0]

job_res_name= 'lalonde_bdhsic_perm'

if not os.path.exists(job_res_name):
    os.makedirs(job_res_name)
for use_dummy_y in [True,False]:
    for m in [100,150,200]:
        for var in [1,2]:
            for est in ['NCE_Q','real_TRE_Q']:
                for sep in [False]:
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
                            'variant': var
                            }
                    pval_list = []
                    for i in range(100):
                        # i=21
                        print('seed: ',i)
                        torch.random.manual_seed(i)
                        perm = torch.randperm(n)[:m]
                        x = X[perm]
                        if use_dummy_y:
                            y = torch.randn_like(x)
                        else:
                            y = Y[perm]
                        z = Z[perm]
                        try:
                            sim_obj = simulation_object_rule_perm(args=args) #Why getting nans?
                            pval, ref = sim_obj.run_data(x,y,z,cat_data)
                        except Exception as e:
                            print (e)
                        pval_list.append(pval)
                    pvals = torch.tensor(pval_list)
                    torch.save({'pvals':pvals,'config':args},f'{job_res_name}/lalonde_pvals_n={m}_{est}_{sep}_kernel={var}_dummy={use_dummy_y}.pt')




