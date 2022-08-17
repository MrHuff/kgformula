import os
import torch
from kgformula.utils import simulation_object_rule_new,simulation_object_rule_perm


job_res_name= 'twins_exp_no_sep_perm'

if not os.path.exists(job_res_name):
    os.makedirs(job_res_name)

X,Y,Z,ind_dict = torch.load('twins.pt')
n = X.shape[0]

for use_dummy_y in [True,False]:
    for m in [5000]:
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
                            # sim_obj = simulation_object_rule_new(args=args) #Why getting nans?
                            sim_obj = simulation_object_rule_perm(args=args) #Why getting nans?
                            pval, ref = sim_obj.run_data(x,y,z,ind_dict)
                        except Exception as e:
                            print (e)
                        pval_list.append(pval)
                    pvals = torch.tensor(pval_list)
                    torch.save({'pvals':pvals,'config':args},f'{job_res_name}/twins_pvals_{est}_{sep}_{m}_{use_dummy_y}_kernel={var}.pt')



