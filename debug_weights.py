from kgformula.networks import *
from kgformula.test_statistics import *
from kgformula.utils import *
import GPUtil
width = 24
layers = 2
val_rate = 0.05
n=5000
i=0
estimator = 'NCE' #NCE_Q, #TRE_Q
data_dir = ''
est_params= {'lr': 1e-4,
               'max_its': 5000,
               'width': width,
               'layers': layers,
               'mixed': False,
               'bs_ratio': 10. / n,
               'val_rate': val_rate,
               'n_sample': 250,
               'criteria_limit': 0.05,
               'kill_counter': 10,
               'kappa': 10,
               'm': n
               }
device = GPUtil.getAvailable(order='memory', limit=8)[0]
torch.cuda.set_device(device)
X, Y, Z, X_q, _w, w_q = torch.load(f'./{data_dir}/data_seed={i}.pt', map_location=f'cuda:{device}')
X_train,X_test = split(X,n)
Z_train,Z_test = split(Z,n)
d = density_estimator(x=X_train, z=Z_train, cuda=True,
                      est_params=est_params, type=estimator, device=device)
w = d.return_weights(X_test, Z_test)
