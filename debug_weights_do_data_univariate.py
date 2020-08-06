from debug_weights import *
from kgformula.utils import experiment_plt_mv,experiment_plt_do

if __name__ == '__main__':
    n=10000 #Increase to 5k or 10k... Should yield more robust results.
    device = GPUtil.getFirstAvailable(order='memory')[0]
    train_factor = 0.5
    var = 1.0
    scale = 0.7
    d_X = 1
    d_Z = 1
    for experiment in [0]:
        if experiment ==0: #Univariate density ratio where the true ratio has a mode, i.e. simplest case?!
            data_path = f'univariate_1000_seeds/ground_truth=H_0_y_a=0.0_y_b=0.0_z_a=0.0_z_b=0.5_cor=0.5_n={n}_seeds=1000_0.9_0.81/data_seed=0.pt'

        X,Y,Z,w_true = torch.load(data_path)
        train_mask = np.random.rand(n)<train_factor
        X_train = X[train_mask,:]
        Z_train = Z[train_mask,:]
        X_test = X[~train_mask,:]
        Z_test = Z[~train_mask,:]

        mse_loss = torch.nn.MSELoss()
        estimator = 'NCE' #gradient fix?
        est_params = {'lr': 1e-4,
                      'max_its': 5000,
                      'width': 32,
                      'layers': 4,
                      'mixed': False,
                      'bs_ratio': 50/n,
                      'kappa': 10,
                      'val_rate': 5e-2,
                      'n_sample': 1000,
                      'criteria_limit': 0.25,
                      'kill_counter': 100,
                      'outputs': [1, 1],
                      'reg_lambda':1e-1,
                      'm':n,
                      'd_X':d_X,
                      'd_Z':d_Z,
                      'latent_dim':16,
                      'depth_u': 2,
                      'depth_v': 2,
                      'IP':False,
                      'scale_x':scale
                      }
        # debug_W(w_true,f'w_true_{experiment}')
        torch.cuda.set_device(device)
        X_train,Z_train,X_test,Z_test =X_train.cuda(),Z_train.cuda(),X_test.cuda(),Z_test.cuda()
        print(X_train.shape,Z_train.shape)
        d = get_w_estimate_and_plot(X_train, Z_train, est_params, estimator, device)
        w_classification = d.return_weights(X_test,Z_test)
        # debug_W(d,f'w_classification_{experiment}')
        with torch.no_grad():
            l = mse_loss(w_true[~train_mask].cuda(),w_classification)/w_true[~train_mask].var()
            print(1-l.item())
        experiment_plt_do(w_true[~train_mask],w_classification,X_test,Z_test,f'Experiment_{experiment}_{estimator}_do_uni_scale={scale}')