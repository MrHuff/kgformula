import pandas as pd
import torch
from kgformula.utils import simulation_object_rule_new

df = pd.read_csv("lalonde.csv")

X = torch.from_numpy(df['treat'].values).unsqueeze(-1).float()
Y = torch.from_numpy(df['re78'].values).unsqueeze(-1).float()
Z = torch.from_numpy(df[['age','education','black','hispanic','married','nodegree','re74','re75']].values).float()

args = {'qdist':2,
        'bootstrap_runs':250,
        'est_params': {'lr': 1e-4, #use really small LR for TRE. Ok what the fuck is going on...
                                                   'max_its': 10,
                                                   'width': 32,
                                                   'layers':3,
                                                   'mixed': False,
                                                   'bs_ratio': 1e-2,
                                                   'val_rate': 0.05,
                                                   'n_sample': 250,
                                                   'criteria_limit': 0.05,
                                                   'kill_counter': 2,
                                                    'kappa':10,
                                                   'm': 6
                                                   },
        'estimator': 'NCE_Q',
        'cuda': True,
        'device': 'cuda:0',
        'n':10000,
        'unique_job_idx':0
        }
sim_obj = simulation_object_rule_new(args=args) #Why getting nans?
print(sim_obj.run_data(X,Y,Z))


