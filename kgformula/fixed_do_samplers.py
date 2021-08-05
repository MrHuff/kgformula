import torch
from torch.distributions import Normal,Beta,Gamma,Exponential,Bernoulli
from scipy.stats import t,gamma,norm
import numpy as np
import warnings
import math
import matplotlib.pyplot as plt
def nCr(n,r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)
def get_sigma(N,cors):
    if cors.min()==cors.max():
        Sigma = torch.ones(*(2, 2))
        Sigma[0, 1] = cors[0]
        Sigma[1, 0] = cors[0]
    else:
        Sigma = torch.stack([cors,torch.sqrt(1-cors**2)],dim=1) #Nx2
    return Sigma

def rfgmCopula(n,d,alpha):
    U_mat = torch.rand(*(n,d-1))
    U = torch.rand(*(n,1))
    B = torch.prod(alpha*torch.ones_like(U_mat)-2*U_mat,dim=1)
    C = torch.sqrt((1+B)**2-4*B*U)
    return torch.div((2*U),1+B+C)

def sim_X(n,dist,theta,d_X=1,phi=1.5):
    if dist==1:
        d = Normal(loc=0,scale=theta*phi)
    elif dist== 4:
        d = Beta(concentration0=theta,concentration1=theta)
    elif dist== 3:
        d = Gamma(concentration=theta*phi,rate=1./torch.exp(torch.tensor(1.))) #log-Expectation is phi=1.0

    else:
        raise Exception("X distribution must be normal (1), beta (4) or gamma (3)")
    return {'data':d.sample((n,d_X)),'density':d.log_prob}

def sim_bin_X(n,d_X=1):
    d = Bernoulli(probs=0.5)
    return {'data':d.sample((n,d_X)),'density':d.log_prob}


def sim_inv_cdf_X(X,dist,theta,phi=1.5):
    if dist==1:
        d = Normal(loc=0,scale=theta*phi)
        X = d.icdf(X)
    elif dist== 4:
        d = Beta(concentration0=theta,concentration1=theta)
        X = d.icdf(X)
    elif dist== 3:
        d = Gamma(concentration=theta*phi,rate=1./torch.exp(torch.tensor(1.))) #log-Expectation is phi=1.0
        X = torch.from_numpy(gamma.ppf(X.numpy(),a=theta*phi,scale=np.exp(1.0))).float()
    else:
        raise Exception("X distribution must be normal (1), beta (4) or gamma (3)")
    return {'data':X,'density':d.log_prob}

def rnormCopula(N,cov):
    if cov.dim()==2:
        m = cov.shape[0]
        mean = torch.zeros(*(N,m))
        L = torch.cholesky(cov)
        samples = mean +  torch.randn_like(mean)@L.t()

    elif cov.dim()==3:
        m = cov[0].shape[0]
        mean = torch.zeros(*(N,m))
        noise = torch.randn_like(mean).unsqueeze(1)
        if torch.cuda.is_available():
            mean = torch.zeros(*(N, m)).cuda()
            noise = noise.cuda()
            cov=cov.cuda()
        L = torch.cholesky(cov,upper=True)
        samples = torch.bmm(noise, L).squeeze() + mean
        samples = samples.cpu().numpy()
    # for i in range(samples.shape[1]):
    #     plt.hist(norm.cdf(samples[:,i]),40)
    #     plt.savefig(f'copula_sanity_{i}.png')
    #     plt.clf()
    return torch.from_numpy(norm.cdf(samples)).float() #Dude make it uniform...

def expit(x):
    return torch.exp(x)/(1+torch.exp(x))

def apply_qdist(inv_wts,q_factor,theta,X):
    d=Normal(0,theta*q_factor)
    q_dens = 0
    for i in range(X.shape[1]):
        q_dens += d.log_prob(X[:,i])
    q_dens = q_dens.exp()
    w_q = inv_wts*q_dens.squeeze()
    X_q  = d.sample((X.shape[0],X.shape[1]))
    return X_q,w_q


def sim_UVW(N,total_d,cor):
    triang_cov = torch.tensor(cor)
    s = torch.eye(total_d)
    sigma = s
    sigma[torch.triu(torch.ones_like(sigma),diagonal=1)==1] = triang_cov
    sigma[torch.tril(torch.ones_like(sigma),diagonal=-1)==1] = triang_cov
    tmp = rnormCopula(N,sigma)
    return tmp

def sim_multivariate_UV(dat, mv_type, par, total_d,ref_dim):
    N = dat.shape[0]
    pars = torch.cat([torch.ones(*(dat.shape[0],1)),dat],dim=1)@par# (N*(x_d+1)@ ((x_d+1)*ref_dim) )  Make depedency on X to copula between Y. Should be N x triangular size.
    pars = pars.unsqueeze(-1).repeat(1,int(ref_dim))
    cors = torch.clip(2*torch.sigmoid(pars)-1,-0.99,0.99)
    s = torch.eye(total_d)
    if mv_type==1:
        sigma = s
        sigma[torch.triu(torch.ones_like(sigma),diagonal=1)==1] = cors[0,:]
        sigma[torch.tril(torch.ones_like(sigma),diagonal=-1)==1] = cors[0,:]
        tmp = rnormCopula(N,sigma)
    else:
        sigma = torch.eye(total_d).unsqueeze(0).repeat(cors.shape[0],1,1)
        sigma[:, torch.triu(torch.ones_like(s), diagonal=1) == 1] = cors
        sigma[:, torch.tril(torch.ones_like(s), diagonal=-1) == 1] = cors
        tmp = rnormCopula(N,sigma)

    dat = torch.cat([dat,tmp],dim=1)
    return dat

def sim_multivariate_XYZ(oversamp,d_Z,n,beta_xy,beta_xz,yz=[0.5,0.0],fam_z=1,fam_x=[1,1],phi=1,theta=1,d_Y=1,d_X=1,fam_y=1):
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    if oversamp < 1:
        warnings.warn("Oversampling rate must be at least 1... changing")
        oversamp = 1
    N = round(oversamp*n)
    if len(yz)==((d_X+d_Z+d_Y)**2-(d_X+d_Z+d_Y))/2:
        dat = sim_UVW(N,d_X+d_Z+d_Y,yz)
        x_tmp = sim_inv_cdf_X(X=dat[:,0:d_X],dist=fam_x[0],theta=theta,phi=phi)
        X = x_tmp['data']
        qden = x_tmp['density']
    else:
        cor = torch.tensor([yz[0]] + [yz[1]] * d_X)
        if yz[1]==0.0: #no dependency on x in cupola
            mv_type=1
            ref_dim = 1
        else:
            mv_type=2
            ref_dim = ((d_Z+d_Y)**2-(d_Z+d_Y))/2
        tmp = sim_X(n=N,dist=fam_x[0],theta=theta,d_X=d_X,phi=phi) #nxd_X
        dat = tmp['data']
        qden = tmp['density']
        dat = sim_multivariate_UV(dat,mv_type,cor,d_Z+d_Y,ref_dim)
        X = dat[:, 0:d_X]
    Y = dat[:, d_X:(d_X + d_Y), ]
    Z = dat[:, (d_X + d_Y):(d_X + d_Y + d_Z)]
    plt.hist(X.numpy(),bins=40)
    plt.savefig('marg_x_viz.png')
    plt.clf()
    a = beta_xy[0]
    b = beta_xy[1]  # Controls X y dependence
    if fam_y==1:
        if torch.is_tensor(b):
            p = Normal(loc=a+X@b,scale=1) #Consider square matrix valued b.
        else:
            p = Normal(loc=a+X*b,scale=1) #Consider square matrix valued b.
    elif fam_y==3:
        if torch.is_tensor(b):
            p = Exponential(rate=torch.exp( (a + X @ b)))  #Incorrect expectation? Consider square matrix valued b.
        else:
            p = Exponential(rate=torch.exp( (a + X * b)))  # Consider square matrix valued b.
    Y = p.icdf(Y) # Change this to exponential...

    if fam_z == 1:
        q = Normal(loc=0, scale=1) #This is still OK
        Z = q.icdf(Z)
    elif fam_z == 3:
        q = Exponential(rate=1) #Bug in code you are not sampling exponentials!!!!!
        Z = q.icdf(Z)
    else:
        raise Exception("fam_z must be 1, 2 or 3")

    beta_xz = torch.tensor(beta_xz).float()
    if beta_xz.dim()<2:
        beta_xz = beta_xz.unsqueeze(-1)
    _x_mu = torch.cat([torch.ones(*(X.shape[0],1)),Z],dim=1) @ beta_xz #XZ dependence (n x (1+d)) matmul (1+d x 1)
    #Do GLM? Poisson Link function...
    # Look at X[:,0] - _x_mu[:,0]. Run KS test on that quantity for distribution with correct variance.
    # repeat for each "column". i.e. X[:,i] - _x_mu[:,i].
    # Think each of X and something to regress on Z. Think of the target distribtion. on what you expect post rejection sampling.

    if fam_x[1] == 4:
        mu = expit(_x_mu)
        d = Beta(concentration1=theta * mu, concentration0=theta * (1 - mu))
    elif fam_x[1] == 1: #Signal to noise ratio
        mu = _x_mu
        d = Normal(loc=mu, scale=theta) #ks -test uses this target distribution. KS-test on  0 centered d with scale phi...
        #might wanna consider d_X d's for more beta_XZ's
    elif fam_x[1] == 3:  # Change
        mu = torch.exp(_x_mu+1.0) #Poisson link func? theta=phi
        d = Gamma(concentration=theta,rate=1./mu ) #Scale everything with 1/mu so that it looks like iid samples from some gamma distribution. Then KS-test on 1/theta
    else:
        raise Exception("fam_x must be 1, 3 or 4")
    wts = torch.zeros(*(X.shape[0],1))
    #To make Rejectoin sampling work, principal eigenvalue of target distribution i.e. d. Should be less than theta.
    p_z  = torch.zeros(*(X.shape[0],1))
    for i in range(d_X):
        _x = X[:,i].unsqueeze(-1)

        # d_samp = d.sample((1,1))
        # plt.hist(X[:,i].squeeze().numpy(),50,color='blue',alpha=0.5)
        # plt.hist(d_samp.squeeze().numpy(),50,color='red',alpha=0.5)
        # plt.savefig(f'density_sanity_check_{i}.png')
        # plt.clf()

        p_cond_z = d.log_prob(_x)
        p_z += p_cond_z
        _prob = p_cond_z- qden(_x)
        wts = wts + _prob
    wts = wts.exp()
    p_z = p_z.exp()
    if fam_x[0] == 1 and fam_x[1] == 1:
        max_ratio_points = mu  * phi / ( phi - 1.)
        normalization = (d_X*d.log_prob(max_ratio_points) - d_X*qden(max_ratio_points)).exp()
    elif fam_x[0] == 3 and fam_x[1] == 3:
        # max_ratio_points =(theta*phi-theta)/(1./torch.exp(torch.tensor(1.))-1./mu)
        # normalization = (d_X*d.log_prob(max_ratio_points) - d_X*qden(max_ratio_points)).exp()
        normalization = wts.max()
    else:
        print("Warning: No analytical solution exist for maximum density ratio using default sample max")
        normalization = wts.max()
    if torch.isnan(wts).all():
        raise Exception("Problem with weights")
    wts_tmp = wts / normalization
    keep_index = (torch.rand_like(wts) < wts_tmp).squeeze()
    inv_wts = 1. / p_z #Variance of the weights seems to blow up when n is large, this also causes problems for the estimator...
    X,Y,Z,inv_wts = X[keep_index,:],Y[keep_index,:],Z[keep_index,:], inv_wts[keep_index]
    return X,Y,Z,inv_wts


def simulate_xyz_mixed_multivariate(n, oversamp,d_Z,beta_xy,beta_xz,yz,seed,d_Y=1,d_X=1,phi=2,theta=2,fam_x=[1,1],fam_z=1,fam_y=1):
    """
    beta_xz has dim (d_Z+1) list
    beta_xy has dim 2 list
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    X,Y,Z,w = sim_mixed_multivariate_XYZ(oversamp=oversamp,
                                   d_Z=d_Z,
                                   n=n,
                                   beta_xy=beta_xy,
                                   beta_xz=beta_xz,
                                   yz=yz,
                                   fam_z=fam_z,
                                   fam_x=fam_x,
                                   phi=phi,
                                   theta=theta,
                                   d_X=d_X,
                                   d_Y=d_Y,
                                   fam_y=fam_y,
                                   )
    while X.shape[0]<n:
        print(f'Undersampled: {X.shape[0]}')
        oversamp = oversamp*1.01
        X_new,Y_new,Z_new, w_new= sim_mixed_multivariate_XYZ(oversamp=oversamp,
                                                       d_Z=d_Z,
                                                       n=n,
                                                       beta_xy=beta_xy,
                                                       beta_xz=beta_xz,
                                                       yz=yz,
                                                       fam_z=fam_z,
                                                       fam_x=fam_x,
                                                       phi=phi,
                                                       theta=theta,
                                                       d_X=d_X,
                                                       d_Y=d_Y,
                                                       fam_y=fam_y,
                                                       )
        X = torch.cat([X,X_new],dim=0)
        Y = torch.cat([Y,Y_new],dim=0)
        Z = torch.cat([Z,Z_new],dim=0)
        w = torch.cat([w,w_new],dim=0)

    print(f'Ok: {X.shape[0]}')

    if X.dim()<2:
        X=X.unsqueeze(-1)
    if Y.dim()<2:
        Y=Y.unsqueeze(-1)
    if Z.dim()<2:
        Z=Z.unsqueeze(-1)

    return X[0:n,:],Y[0:n,:],Z[0:n,:],w[0:n].squeeze()


def sim_mixed_multivariate_XYZ(oversamp,d_Z,n,beta_xy,beta_xz,yz=[0.5,0.0],fam_z=1,fam_x=[1,1],phi=1,theta=1,d_Y=1,d_X=1,fam_y=1):
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    if oversamp < 1:
        warnings.warn("Oversampling rate must be at least 1... changing")
        oversamp = 1
    N = round(oversamp*n)

    cor = torch.tensor([yz[0]] + [yz[1]] * d_X)
    if yz[1]==0.0: #no dependency on x in cupola
        mv_type=1
        ref_dim = 1
    else:
        mv_type=2
        ref_dim = ((d_Z+d_Y)**2-(d_Z+d_Y))/2
    tmp = sim_X(n=N,dist=fam_x[0],theta=theta,d_X=d_X//2,phi=phi) #nxd_X
    tmp_2 = sim_bin_X(n=N,d_X=d_X//2)

    dat = torch.cat([tmp['data'],tmp_2['data']],dim=1)
    qden_1 = tmp['density']
    qden_2 = tmp_2['density']
    dat = sim_multivariate_UV(dat,mv_type,cor,d_Z+d_Y,ref_dim)
    X = dat[:, 0:d_X]

    #try binarizing X at this point
    Y = dat[:, d_X:(d_X + d_Y), ]
    Z = dat[:, (d_X + d_Y):(d_X + d_Y + d_Z)]
    plt.hist(X.numpy(),bins=40)
    plt.savefig('marg_x_viz.png')
    plt.clf()
    a = beta_xy[0]
    b = beta_xy[1]  # Controls X y dependence
    if fam_y==1:
        if torch.is_tensor(b):
            p = Normal(loc=a+X@b,scale=1) #Consider square matrix valued b.
        else:
            p = Normal(loc=a+X*b,scale=1) #Consider square matrix valued b.
    elif fam_y==3:
        if torch.is_tensor(b):
            p = Exponential(rate=torch.exp( (a + X @ b)))  #Incorrect expectation? Consider square matrix valued b.
        else:
            p = Exponential(rate=torch.exp( (a + X * b)))  # Consider square matrix valued b.
    Y = p.icdf(Y) # Change this to exponential...

    if fam_z == 1:
        q = Normal(loc=0, scale=1) #This is still OK
        Z = q.icdf(Z)
    elif fam_z == 3:
        q = Exponential(rate=1) #Bug in code you are not sampling exponentials!!!!!
        Z = q.icdf(Z)
    else:
        raise Exception("fam_z must be 1, 2 or 3")

    beta_xz = torch.tensor(beta_xz).float()
    if beta_xz.dim()<2:
        beta_xz = beta_xz.unsqueeze(-1)
    _x_mu = torch.cat([torch.ones(*(X.shape[0],1)),Z],dim=1) @ beta_xz #XZ dependence (n x (1+d)) matmul (1+d x 1)
    _p = torch.sigmoid(_x_mu) #XZ dependence (n x (1+d)) matmul (1+d x 1)
    #Do GLM? Poisson Link function...
    # Look at X[:,0] - _x_mu[:,0]. Run KS test on that quantity for distribution with correct variance.
    # repeat for each "column". i.e. X[:,i] - _x_mu[:,i].
    # Think each of X and something to regress on Z. Think of the target distribtion. on what you expect post rejection sampling.

    if fam_x[1] == 4:
        mu = expit(_x_mu)
        d = Beta(concentration1=theta * mu, concentration0=theta * (1 - mu))
    elif fam_x[1] == 1: #Signal to noise ratio
        mu = _x_mu
        d = Normal(loc=mu, scale=theta) #ks -test uses this target distribution. KS-test on  0 centered d with scale phi...
        #might wanna consider d_X d's for more beta_XZ's
    elif fam_x[1] == 3:  # Change
        mu = torch.exp(_x_mu+1.0) #Poisson link func? theta=phi
        d = Gamma(concentration=theta,rate=1./mu ) #Scale everything with 1/mu so that it looks like iid samples from some gamma distribution. Then KS-test on 1/theta
    else:
        raise Exception("fam_x must be 1, 3 or 4")
    d_bin = Bernoulli(probs=_p)
    wts = torch.zeros(*(X.shape[0],1))
    #To make Rejectoin sampling work, principal eigenvalue of target distribution i.e. d. Should be less than theta.
    p_z  = torch.zeros(*(X.shape[0],1))
    for i in range(d_X//2):
        _x = X[:,i].unsqueeze(-1)
        p_cond_z = d.log_prob(_x)
        p_z += p_cond_z
        _prob = p_cond_z- qden_1(_x)
        wts = wts + _prob
    for i in range(d_X//2,d_X):
        _x = X[:,i].unsqueeze(-1)
        p_cond_z = d_bin.log_prob(_x)
        p_z += p_cond_z
        _prob = p_cond_z- qden_2(_x)
        wts = wts + _prob
    wts = wts.exp()
    p_z = p_z.exp()
    normalization = wts.max()
    if torch.isnan(wts).all():
        raise Exception("Problem with weights")
    wts_tmp = wts / normalization
    keep_index = (torch.rand_like(wts) < wts_tmp).squeeze()
    inv_wts = 1. / p_z #Variance of the weights seems to blow up when n is large, this also causes problems for the estimator...
    X,Y,Z,inv_wts = X[keep_index,:],Y[keep_index,:],Z[keep_index,:], inv_wts[keep_index]
    return X,Y,Z,inv_wts


def simulate_xyz_multivariate(n, oversamp,d_Z,beta_xy,beta_xz,yz,seed,d_Y=1,d_X=1,phi=2,theta=2,fam_x=[1,1],fam_z=1,fam_y=1):
    """
    beta_xz has dim (d_Z+1) list
    beta_xy has dim 2 list
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    X,Y,Z,w = sim_multivariate_XYZ(oversamp=oversamp,
                                   d_Z=d_Z,
                                   n=n,
                                   beta_xy=beta_xy,
                                   beta_xz=beta_xz,
                                   yz=yz,
                                   fam_z=fam_z,
                                   fam_x=fam_x,
                                   phi=phi,
                                   theta=theta,
                                   d_X=d_X,
                                   d_Y=d_Y,
                                   fam_y=fam_y,
                                   )
    while X.shape[0]<n:
        print(f'Undersampled: {X.shape[0]}')
        oversamp = oversamp*1.01
        X_new,Y_new,Z_new, w_new= sim_multivariate_XYZ(oversamp=oversamp,
                                                       d_Z=d_Z,
                                                       n=n,
                                                       beta_xy=beta_xy,
                                                       beta_xz=beta_xz,
                                                       yz=yz,
                                                       fam_z=fam_z,
                                                       fam_x=fam_x,
                                                       phi=phi,
                                                       theta=theta,
                                                       d_X=d_X,
                                                       d_Y=d_Y,
                                                       fam_y=fam_y,
                                                       )
        X = torch.cat([X,X_new],dim=0)
        Y = torch.cat([Y,Y_new],dim=0)
        Z = torch.cat([Z,Z_new],dim=0)
        w = torch.cat([w,w_new],dim=0)

    print(f'Ok: {X.shape[0]}')

    if X.dim()<2:
        X=X.unsqueeze(-1)
    if Y.dim()<2:
        Y=Y.unsqueeze(-1)
    if Z.dim()<2:
        Z=Z.unsqueeze(-1)

    return X[0:n,:],Y[0:n,:],Z[0:n,:],w[0:n].squeeze()




