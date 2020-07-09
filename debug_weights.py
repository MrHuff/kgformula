from kgformula.test_statistics import  density_estimator,HSIC_independence_test,hsic_sanity_check_w
from kgformula.utils import simulation_object,get_density_plot
import torch
import GPUtil
from kgformula.utils import get_w_estimate_and_plot,debug_W
import seaborn as sns; sns.set()
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal


#Generate multivariate data for z and for x. Make sure to have a ground truth...
#Think about ESS
# Try KMM again?!

experiment = -1

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
    if experiment ==-1:
        cov = torch.tensor([[2., 1.], [1., 2.]])
        print(cov)
        M, samples, log_prob = generate_experiment_data(MultivariateNormal, torch.zeros(2), cov, 1000)
        dist_a,dist_b,_,_,_,_ = generate_experiment_data_2dist(Normal,torch.tensor([0.]),torch.tensor([0.]),torch.tensor([2.]),torch.tensor([2.]),1000)
        X,Z = torch.unbind(samples,dim=1)
        w_true = (log_prob-(dist_a.log_prob(X)+dist_b.log_prob(Z))).exp()
        X = X.unsqueeze(-1)
        Z = Z.unsqueeze(-1)


    elif experiment==0:
        pass
    elif experiment==1:
        pass
    elif experiment==2:
        pass
    elif experiment==3:
        pass

    mse_loss = torch.nn.MSELoss()
    estimator = 'TRE'
    device = GPUtil.getFirstAvailable(order='memory')[0]
    est_params = {'lr': 1e-4,
                  'max_its': 5000,
                  'width': 128,
                  'layers': 4,
                  'mixed': False,
                  'bs_ratio': 1e-2,
                  'kappa': 10,
                  'val_rate': 1e-2,
                  'n_sample': 1000,
                  'criteria_limit': 0.25,
                  'kill_counter': 10,
                  'depth_main': 3,
                  'depth_task': 3,
                  'outputs': [1, 1, 1,1,1]}
    get_density_plot(w_true, X, Z)
    debug_W(w_true,'w_true')
    torch.cuda.set_device(device)
    X,Z =X.cuda(),Z.cuda()
    w_classification = get_w_estimate_and_plot(X, Z, est_params, estimator, device)
    debug_W(w_classification,'w_classification')
    with torch.no_grad():
        l = mse_loss(w_true.cuda(),w_classification)/w_true.var()
        print(1-l.item())