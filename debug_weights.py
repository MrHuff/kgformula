import torch
import GPUtil
from kgformula.utils import get_w_estimate_and_plot,experiment_plt,debug_W
import seaborn as sns; sns.set()
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
import numpy as np
#Generate multivariate data for z and for x. Make sure to have a ground truth...
#Think about ESS
# Try KMM again?!


def generate_experiment_data(dist,mean,cov,n=1000):
    M = dist(mean,cov)
    samples = M.sample(torch.Size([n]))
    log_prob = M.log_prob(samples)
    return M,samples,log_prob

def generate_experiment_data_2dist(dist,mean_A,mean_B,cov_A,cov_B,n=1000):
    A_dist,sample_A,log_prob_A = generate_experiment_data(dist,mean_A,cov_A,n)
    B_dist,sample_B,log_prob_B = generate_experiment_data(dist,mean_B,cov_B,n)
    return A_dist,B_dist,sample_A,sample_B,log_prob_A,log_prob_B

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
    #Test on actual statistic
    #Doesnt seem to capture tails

    var = 1.
    scale = 0.7
    for experiment in [0]:
        if experiment ==0: #Univariate density ratio where the true ratio has a mode, i.e. simplest case?!
            cov = torch.tensor([[var,0.1], [0.1, var]])
        elif experiment==1:#Univariate density ratio where the true ratio has no mode, i.e. univariate w distribution
            cov = torch.tensor([[var, 0.2], [0.2, var]]) #Higher covariance should be easier...
        elif experiment==2: ##Univariate density ratio where the true ratio has an inflated mode, i.e. mimics the multivariate case where things go wrong.
            cov = torch.tensor([[var, 0.3], [0.3, var]])
        elif experiment==3:
            cov = torch.tensor([[var, 0.4], [0.4, var]])
        elif experiment==4:
            cov = torch.tensor([[var, 0.5], [0.5, var]])
        elif experiment==5:
            cov = torch.tensor([[var, 0.6], [0.6, var]])
        if experiment == 6:  # Univariate density ratio where the true ratio has a mode, i.e. simplest case?!
            cov = torch.tensor([[var, 0.05], [0.05, var]])


        #eigen values from joint i.e. cov variable must be smaller than product of margins. Try simulating such a case...
        #Be careful to what we pass to as training data, ensure only difference is dependencce. So decouple properly.
        #How to do so doing so while ensuring all properties... #Rejection sampling approach to simulate what we actually want to
        #Change the loss function: plausibility of product of the margins.
        # Multiply weights with function of x and z to cap the expectation of weights
        # Assure independence of the fitted data... train-test setup
        # Actually try multivariate distributions...
        # Apply e(-x^2)e(-z^2) to weights... Choose f(x)f(z) smartly such that the expectation is still valid...
        #Should be able to factor in directly...
        #Try q(x) as margins. additional parametrization for q_\theta(x). Ok convince ourselves first this is OK... We have convinced ourselves this is OK (disclaimer potentially OK)
        #Just rescale x it down-> such that eigenvalue matrices are OK. 
        # Triple independence, first weights trained on separate dataset than covariance estimator
        # second traning set divided into 2 sets for joint samples and product of the margins...
        M, samples, log_prob = generate_experiment_data(MultivariateNormal, torch.zeros(2), cov, n)
        dist_a, dist_b, _, _, _, _ = generate_experiment_data_2dist(Normal, torch.tensor([0.]), torch.tensor([0.]),
                                                                    torch.tensor([var ** 0.5]),
                                                                    torch.tensor([var ** 0.5]), n)

        X, Z = torch.unbind(samples, dim=1)
        w_true = (-log_prob + (dist_a.log_prob(X) + dist_b.log_prob(Z))).exp()
        X = X.unsqueeze(-1)
        Z = Z.unsqueeze(-1)
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
                      'd_X':1,
                      'd_Z':1,
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
        experiment_plt(w_true[~train_mask],w_classification,X_test,Z_test,f'Experiment_{experiment}_{estimator}_scale={scale}',var,M,dist_a,dist_b,d.model)