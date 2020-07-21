from kgformula.test_statistics import  density_estimator,HSIC_independence_test,hsic_sanity_check_w
from kgformula.utils import simulation_object,get_density_plot
import torch
import GPUtil
from kgformula.utils import get_w_estimate_and_plot,experiment_plt,debug_W
import seaborn as sns; sns.set()
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from matplotlib import pyplot as plt
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
    n=1000
    device = GPUtil.getFirstAvailable(order='memory')[0]
    for experiment in [0,1,2]:
        if experiment ==0: #Univariate density ratio where the true ratio has a mode, i.e. simplest case?!
            var = 1.
            cov = torch.tensor([[var,0.1], [0.1, var]])
            print(cov)
            M, samples, log_prob = generate_experiment_data(MultivariateNormal, torch.zeros(2), cov, n)
            dist_a,dist_b,_,_,_,_ = generate_experiment_data_2dist(Normal,torch.tensor([0.]),torch.tensor([0.]),torch.tensor([var**0.5]),torch.tensor([var**0.5]),n)
            X,Z = torch.unbind(samples,dim=1)
            w_true = (-log_prob+(dist_a.log_prob(X)+dist_b.log_prob(Z))).exp()
            X = X.unsqueeze(-1)
            Z = Z.unsqueeze(-1)

        elif experiment==1:#Univariate density ratio where the true ratio has no mode, i.e. univariate w distribution
            var = 1.
            cov = torch.tensor([[var, 0.5], [0.5, var]])
            print(cov)
            M, samples, log_prob = generate_experiment_data(MultivariateNormal, torch.zeros(2), cov, n)
            dist_a, dist_b, _, _, _, _ = generate_experiment_data_2dist(Normal, torch.tensor([0.]), torch.tensor([0.]),
                                                                        torch.tensor([var**0.5]), torch.tensor([var**0.5]), n)
            X, Z = torch.unbind(samples, dim=1)
            w_true = (-log_prob + (dist_a.log_prob(X) + dist_b.log_prob(Z))).exp()
            X = X.unsqueeze(-1)
            Z = Z.unsqueeze(-1)
        elif experiment==2: ##Univariate density ratio where the true ratio has an inflated mode, i.e. mimics the multivariate case where things go wrong.
            var = 0.75
            cov = torch.tensor([[var, 0.25], [0.25, var]])
            M, samples, log_prob = generate_experiment_data(MultivariateNormal, torch.zeros(2), cov, n)
            dist_a, dist_b, _, _, _, _ = generate_experiment_data_2dist(Normal, torch.tensor([0.]), torch.tensor([0.]),
                                                                        torch.tensor([var**0.5]), torch.tensor([var**0.5]), n)
            X, Z = torch.unbind(samples, dim=1)
            w_true = (-log_prob + (dist_a.log_prob(X) + dist_b.log_prob(Z))).exp()
            X = X.unsqueeze(-1)
            Z = Z.unsqueeze(-1)
        elif experiment==3:
            pass
        elif experiment==4:
            pass



        #Might be a signflip somewhere
        #Do some QP for KMM...
        #get a new plot which shows whether weights are exploding or not. Contour plots for more debugging. What do I expect the plot to look like?!
        #KMM QP or parametric estimator. Binary X etc. Read up on A12. Advanced Simulation Methods
        #Might want to use parametric methods.
        mse_loss = torch.nn.MSELoss()
        estimator = 'kmm_qp' #gradient fix?
        # get_density_plot(w_true, X, Z,f'true_w_{experiment}')
        est_params = {'lr': 1e-4,
                      'max_its': 2500,
                      'width': 32,
                      'layers': 2,
                      'mixed': False,
                      'bs_ratio': 100/n,
                      'kappa': 10,
                      'val_rate': 1e-2,
                      'n_sample': 1000,
                      'criteria_limit': 0.25,
                      'kill_counter': 10,
                      'depth_main': 1,
                      'depth_task': 1,
                      'outputs': [1, 1,1,1],
                      'reg_lambda':1e-1,
                      'm':n}
        # debug_W(w_true,f'w_true_{experiment}')
        torch.cuda.set_device(device)
        X,Z =X.cuda(),Z.cuda()
        print(X.shape,Z.shape)
        d = get_w_estimate_and_plot(X, Z, est_params, estimator, device,f'experiment_w_{experiment}')
        w_classification = d.return_weights()
        # debug_W(d,f'w_classification_{experiment}')
        with torch.no_grad():
            l = mse_loss(w_true.cuda(),w_classification)/w_true.var()
            print(1-l.item())
        experiment_plt(w_true,w_classification,X,Z,f'Experiment_{experiment}_{estimator}',var,M,dist_a,dist_b,d.model)