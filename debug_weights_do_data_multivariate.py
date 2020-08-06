from debug_weights import *
from kgformula.utils import experiment_plt_mv

if __name__ == '__main__':
    n=10000 #Increase to 5k or 10k... Should yield more robust results.
    device = GPUtil.getFirstAvailable(order='memory')[0]
    train_factor = 0.5
    #Go back to AUC should prolly be better than 0.5... Looks like my classifier is stuck.
    #NCE should work
    # Know the true density ratio, figure out "optimal classification accuracy" I should get close
    #Log loss going down as well
    #get a log loss plot
    #get an auc plot
    #0.1, 0.2, -> 0.7. Where does it break?
    #NCE density ratio estimation
    #MINE
    #Representation Learning with Contrastive Predictive Coding go to scholar. Check Gutmann else.
    #Mutual information probably no...
    var = 1.0
    scale=0.5
    for experiment in [0,1,2]:
        if experiment ==0: #Univariate density ratio where the true ratio has a mode, i.e. simplest case?!
            cov_val = 0.1
            cov_prod = torch.tensor([[var,cov_val,cov_val],[cov_val,var,cov_val],[cov_val,cov_val,var] ])
            cov_structure = torch.ones_like(cov_prod)*cov_val
            cov_joint = torch.cat([torch.cat([cov_prod,cov_structure],dim=1),torch.cat([cov_structure,cov_prod],dim=1)],dim=0)
            d_X,d_Z = 3,3
        if experiment ==1: #Univariate density ratio where the true ratio has a mode, i.e. simplest case?!
            cov_val = 0.25
            cov_prod = torch.tensor([[var,cov_val,cov_val],[cov_val,var,cov_val],[cov_val,cov_val,var] ])
            cov_structure = torch.ones_like(cov_prod)*cov_val
            cov_joint = torch.cat([torch.cat([cov_prod,cov_structure],dim=1),torch.cat([cov_structure,cov_prod],dim=1)],dim=0)
            d_X,d_Z = 3,3

        if experiment ==2: #Univariate density ratio where the true ratio has a mode, i.e. simplest case?!
            cov_val = 0.5
            cov_prod = torch.tensor([[var,cov_val,cov_val],[cov_val,var,cov_val],[cov_val,cov_val,var] ])
            cov_structure = torch.ones_like(cov_prod)*cov_val
            cov_joint = torch.cat([torch.cat([cov_prod,cov_structure],dim=1),torch.cat([cov_structure,cov_prod],dim=1)],dim=0)
            d_X,d_Z = 3,3

        M, samples, log_prob = generate_experiment_data(MultivariateNormal, torch.zeros(d_X+d_Z), cov_joint, n)
        dist_a, dist_b, _, _, _, _ = generate_experiment_data_2dist(MultivariateNormal, torch.zeros(d_X),torch.zeros(d_Z),
                                                                    cov_prod,
                                                                    cov_prod, n)
        X = samples[:,:d_X]
        Z = samples[:,d_X:(d_X+d_Z)]
        w_true = (-log_prob + (dist_a.log_prob(X) + dist_b.log_prob(Z))).exp()
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
                      'layers': 2,
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
        experiment_plt_mv(w_true[~train_mask],w_classification,X_test,Z_test,f'Experiment_{experiment}_{estimator}_MV_scale={scale}',var,M,dist_a,dist_b,d.model)