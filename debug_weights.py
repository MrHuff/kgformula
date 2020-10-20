from kgformula.utils import *
import GPUtil


if __name__ == '__main__':
    width = 16
    layers = 2
    val_rate = 0.05
    n=5000
    i=0
    estimator = 'TRE_Q' #NCE_Q, #TRE_Q
    q=1.0
    beta_xz = 0.5
    data_dir =f'univariate_100_seeds/Q={q}_gt=H_0_y_a=0.0_y_b=0.0_z_a=0.0_z_b={beta_xz}_cor=0.5_n=10000_seeds=100_2.0_2.0/'
    est_params= {'lr': 1e-4,
                   'max_its': 5000,
                   'width': width,
                   'layers': layers,
                   'mixed': False,
                   'bs_ratio': 0.01,
                   'val_rate': val_rate,
                   'n_sample': 250,
                   'criteria_limit': 0.05,
                   'kill_counter': 10,
                   'kappa': 1,
                   'm': n,
                    'qdist':1,
                    'qdist_param':{'q_fac':0.8}
                   }
    device = GPUtil.getAvailable(order='memory', limit=8)[0]
    torch.cuda.set_device(device)
    X, Y, Z, X_q, _w, w_q = torch.load(f'./{data_dir}/data_seed={i}.pt', map_location=f'cuda:{device}')
    X_train,X_test = split(X,n)
    Z_train,Z_test = split(Z,n)
    d = density_estimator(x=X_train, z=Z_train, cuda=True,
                          est_params=est_params, type=estimator, device=device)
    w = d.return_weights(X_test, Z_test)
