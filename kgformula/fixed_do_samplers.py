import torch
from torch.distributions import Normal,Beta,Gamma,Exponential
from pycopula.copula import ArchimedeanCopula
from pycopula.simulation import simulate
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
        d = Gamma(concentration=1,rate=1/theta) #Exponential theta
    else:
        raise Exception("X distribution must be normal (1), beta (4) or gamma (3)")
    return {'data':d.sample((n,d_X)),'density':d.log_prob}

def rnormCopula(N,cov):

    if cov.dim()==3:
        pass
    else:
        m = cov.shape[0]
        mean = torch.zeros(*(N,m))
        L = torch.cholesky(cov)
        samples = mean +  torch.randn_like(mean)@L.t()
        # for i in range(samples.shape[1]):
        #     plt.hist(norm.cdf(samples[:,i].numpy()),100)
        #     plt.show()
        #     plt.clf()
        #     print(samples[:,i].std())
    return torch.from_numpy(norm.cdf(samples.numpy())).float()

def rnormCopula2(n=100,mean = torch.zeros(*(2,1)),cov=torch.eye(2),df=1):
    if cov.shape == torch.Size([2,2]):
        l = torch.cholesky(cov)
        M = mean.t() #nx2
        M = M.repeat(n,1)
        samples = M + torch.randn(*(n,2))@l.t()
        # plt.hist(samples[:, 0].numpy(),bins=100)
        # plt.show()
        # plt.clf()
        # print("Y std", torch.std(samples[:, 0]))
        # plt.hist(samples[:, 1].numpy(),bins=100)
        # plt.show()
        # plt.clf()
        # print("Z std", torch.std(samples[:, 1]))
        # Y = samples[:, 0]
        # Z = samples[:,1]
        # print("cor",np.corrcoef(Y.numpy(),Z.numpy()))
    else:
        M = mean.t()
        M = M.repeat(n,1)
        v = torch.randn(n)
        samples = M + torch.stack([v,v*cov[:,0]+torch.randn(n)*cov[:,1]],dim=1)
    return torch.from_numpy(norm.cdf(samples.numpy())).float()

def expit(x):
    return torch.exp(x)/(1+torch.exp(x))

def sim_UV(dat,fam,par,par2):
    if not fam in [1,2,3,4,5,6,11]:
        raise Exception("family not supported")
    N = dat.shape[0]
    if par.shape[0]==2:
        pars = torch.cat([torch.ones_like(dat),dat],dim=1)@par #Don't understand this part.  I think this is a Nx1 matrix?
        pars = pars.squeeze()
    else:
        pars = par #par is scalar

    if fam in [1,2]:
        cors = 2*expit(pars)-1
    elif fam in [3]:
        cors = torch.exp(pars)-1
    elif fam in [4,6]:
        cors = torch.exp(pars)+1
    elif fam in [5]:
        cors = pars
    else:
        cors = 0

    if fam in [1]:
        Sigma = get_sigma(N,cors)
        tmp = rnormCopula2(N,cov=Sigma)
    elif fam in [2]:
        Sigma = get_sigma(N,cors)
        tmp = rnormCopula2(N,cov=Sigma,df=par2)
    elif fam in [11]:
        tmp = rfgmCopula(N,d=2,alpha=cors)
    elif fam in [3,4,5,6]:
        if fam==3:
            copula = ArchimedeanCopula(dim=2,family='clayton')
        elif fam==4:
            copula = ArchimedeanCopula(dim=2,family='gumbel')
        elif fam==5:
            copula = ArchimedeanCopula(dim=2,family='frank')
        elif fam==6:
            copula = ArchimedeanCopula(dim=2,family='joe')
        else:
            raise Exception('Invalid cupola specified')
        if cors.shape[0]==N:
            samples = []
            for i in range(N):
                copula.set_parameter(theta=cors[i])
                s = simulate(copula,1)
                samples.append(s)
            tmp = torch.tensor(samples).float()
        else:
            copula.set_parameter(theta=cors)
            tmp = torch.from_numpy(simulate(copula,N)).float()
    dat = torch.cat([dat,tmp],dim=1)
    return dat

def apply_qdist(inv_wts,q_factor,theta,X):
    d=Normal(0,theta*q_factor)
    q_dens = 0
    for i in range(X.shape[1]):
        q_dens += d.log_prob(X[:,i])
    q_dens = q_dens.exp()
    w_q = inv_wts*q_dens.squeeze()
    X_q  = d.sample((X.shape[0],X.shape[1]))
    return X_q,w_q

def sim_XYZ(n, beta, cor, phi=1, theta=1, par2=1,fam=1, fam_x=[1,1], fam_y=1, fam_z=1,oversamp = 10):
    # q_fac * theta < theta < phi * theta
    if oversamp<1:
        warnings.warn("Oversampling rate must be at least 1... changing")
        oversamp=1

    if type(cor) is not list: #cor controls x xz relation!
        cor = torch.tensor([cor,0]).unsqueeze(-1)
    else:
        cor = torch.tensor(cor).unsqueeze(-1)

    N = round(oversamp*n)
    tmp = sim_X(N,fam_x[0],theta,d_X=1,phi=phi)
    dat = tmp['data']
    qden = tmp['density']
    # plt.hist(dat.numpy(),bins=100)
    # plt.show()

  ## add in extra columns for Y and Zs
  ## get Copula value
    dat = sim_UV(dat, fam, cor, par2) #ZY depedence
    # plt.hist(dat[:,1].numpy(),bins=100)
    # plt.show()
    # plt.clf()
    # plt.hist(dat[:, 2].numpy(),bins=100)
    # plt.show()
    # plt.clf()

    a = beta['y'][0]
    b = beta['y'][1] #Controls X y dependence
    if fam_y==1: #Do XY depdendence
        p = Normal(loc=a+b*dat[:,0],scale=1)
        dat[:,1] = p.icdf(dat[:,1])
    elif fam_y==2:
        dat[:, 1] = torch.from_numpy(t.ppf(dat[:,1].numpy(),df=par2)) + a + b*dat[:,0]
    elif fam_y == 3:
        p = Exponential(rate=1/(a+b*dat[:,0]).exp())
        dat[:,1] = p.icdf(dat[:,1])
    else:
        raise Exception("fam_y must be 1, 2 or 3")
    if fam_z==1:
        q = Normal(loc=0,scale=1)
        dat[:,2] = q.icdf(dat[:,2])
    elif fam_z==2:
        dat[:, 2] = torch.from_numpy(t.ppf(dat[:,2].numpy(),df=par2))
    elif fam_z == 3:
        q = Exponential(rate=1)
        dat[:,2] = q.icdf(dat[:,2])
    else:
        raise Exception("fam_z must be 1, 2 or 3")
    X = torch.stack([torch.ones_like(dat[:, 2]), dat[:, 2]],dim=1) @ torch.tensor(beta['z']) #XZ dependence
    if len(fam_x) == 1:
        fam_x = [fam_x[0],fam_x[0]]
    if fam_x[1] == 4:
        mu = expit(X)
        d = Beta(concentration1=theta*mu,concentration0=theta*(1-mu))
    elif fam_x[1]==1: #do p(x|z) = N(mu,phi)
        mu = X
        d = Normal(loc = mu,scale = theta) #The drastic change DR behaviour is dodgy, find out why and debug this.
    elif fam_x[1]==3: #Change
        mu = torch.exp(X)
        d = Gamma(rate=1/(mu*phi),concentration=1/theta)
    else:
        raise Exception("fam_x must be 1, 3 or 4")
    ## reject samples based on value of odds ratio, induces dependence between X and Z
    p_cond_z = d.log_prob(dat[:, 0])
    wts = (p_cond_z-qden(dat[:, 0])).exp()

    if fam_x[0]==1 and fam_x[1]==1:
        max_ratio_points = mu*theta*phi/(theta*phi-theta)
        normalization = (d.log_prob(max_ratio_points)-qden(max_ratio_points)).exp()
    else:
        print("Warning: No analytical solution exist for maximum density ratio using defautl sample max")
        normalization = wts.max()
    if torch.isnan(wts).all():
        raise Exception("Problem with weights")
    wts_tmp = wts/normalization
    inv_wts=1/p_cond_z.exp() ###
    keep_index = torch.rand_like(wts)<wts_tmp
    dat = dat[keep_index,:]
    #inv_wts multiply by some q_density(X) with smaller variance applied, X=dat[:,0], X_q ~ qden make sure sizes match...
    return dat,inv_wts[keep_index],p_cond_z[keep_index].exp()

def simulate_xyz_univariate(n, beta, cor, phi=2, theta=4, par2=1, fam=1, fam_x=[1, 1], fam_y=1, fam_z=1, oversamp = 10, seed=1,q_factor=0.5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    data,w,p_densities = sim_XYZ(n, beta, cor, phi,theta, par2,fam, fam_x, fam_y, fam_z,oversamp)
    while data.shape[0]<n:
        print(f'Undersampled: {data.shape[0]}')
        oversamp = n/data.shape[0]*1.5
        data_new,w_new,p_densities, = sim_XYZ(n, beta, cor, phi, theta, par2, fam, fam_x, fam_y, fam_z, oversamp)
        data = torch.cat([data,data_new])
        w = torch.cat([w,w_new])
    else:
        print(f'Ok: {data.shape[0]}')
        data = data[0:n,:]

    return data[:,0].unsqueeze(-1),data[:,1].unsqueeze(-1),data[:,2].unsqueeze(-1),w[0:n],p_densities[0:n]

def sample_naive_multivariate(n,d_X,d_Z,d_Y,beta_xz,beta_xy,seed):
    torch.manual_seed(seed)
    Z = torch.randn(n,d_Z)
    X = beta_xz*Z[:,0:d_X]+(1+beta_xz)*torch.randn(n,d_X)
    Y = beta_xy*X[:,0:d_Y]**3+0.1*torch.randn(n,d_Y)+beta_xy/3*Z[:,0:d_Y]**3
    w = torch.ones(n,1)
    return X,Y,Z,w

def sim_multivariate_UV(dat,fam,par,d_z):
    if not fam in [1,3,4,5,6]:
        raise Exception("family not supported")

    N = dat.shape[0]
    pars = torch.cat([torch.ones(*(dat.shape[0],1)),dat],dim=1)@par #Don't understand this part.  I think this is a Nx1 matrix?
    if fam in [1,2]:
        cors = 2*expit(pars)-1
    elif fam in [3]:
        cors = torch.exp(pars)-1
    elif fam in [4,6]:
        cors = torch.exp(pars)+1
    elif fam in [5]:
        cors = pars
    else:
        cors = 0
    if fam in [1]:
        if all(pars[0,:]==pars[-1,:]):
            sigma = torch.eye(d_z)
            sigma[torch.triu(torch.ones_like(sigma),diagonal=1)==1] = pars[0,:]
            sigma[torch.tril(torch.ones_like(sigma),diagonal=-1)==1] = pars[0,:]
            tmp = rnormCopula(N,sigma)
        else:
            sigma = torch.eye(d_z).unsqueeze(-1).repeat(1,1,pars.shape[0])
            tmp = rnormCopula(N,sigma)

    elif fam in [3,4,5,6]:
        if fam==3:
            copula = ArchimedeanCopula(dim=d_z,family='clayton')
        elif fam==4:
            copula = ArchimedeanCopula(dim=d_z,family='gumbel')
        elif fam==5:
            copula = ArchimedeanCopula(dim=d_z,family='frank')
        elif fam==6:
            copula = ArchimedeanCopula(dim=d_z,family='joe')
        else:
            raise Exception('Invalid cupola specified')
        if cors.shape[0]==N:
            samples = []
            for i in range(N):
                copula.set_parameter(theta=cors[i])
                s = simulate(copula,1)
                samples.append(s)
            tmp = torch.tensor(samples).float()
        else:
            copula.set_parameter(theta=cors)
            tmp = torch.from_numpy(simulate(copula,N)).float()
    dat = torch.cat([dat,tmp],dim=1)
    return dat

def sim_multivariate_XYZ(oversamp,d_Z,n,beta_xy,beta_xz,yz,seed,par2=1,fam_z=1,fam_x=[1,1],phi=1,theta=1,d_Y=1,d_X=1,q_fac=1.0):
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    if oversamp < 1:
        warnings.warn("Oversampling rate must be at least 1... changing")
        oversamp = 1

    ref_dim = nCr(d_Z+d_Y,2)
    if type(yz) is not list:  # cor controls x xz relation!
        cor = torch.tensor([yz]+[0]*d_X).unsqueeze(-1)
    else:
        cor = torch.tensor(yz).unsqueeze(-1)
    cor = torch.cat([cor for i in range(ref_dim)],dim=1)
    N = round(oversamp*n)
    tmp = sim_X(N,fam_x[0],theta,d_X=d_X,phi=phi) #nxd_X
    dat = tmp['data']
    qden = tmp['density']
    dat = sim_multivariate_UV(dat,1,cor,d_Z+d_Y)
    a = beta_xy[0]
    b = beta_xy[1]  # Controls X y dependence

    X = dat[:,0:d_X]
    Y = dat[:,d_X:(d_X+d_Y),]
    Z = dat[:,(d_X+d_Y):(d_X+d_Y+d_Z)]

    #Make Y normal!
    if torch.is_tensor(b):
        p = Normal(loc=a+X@b,scale=1) #Consider square matrix valued b.
    else:
        p = Normal(loc=a+X*b,scale=1) #Consider square matrix valued b.
    Y = p.icdf(Y)

    if fam_z == 1:
        q = Normal(loc=0, scale=1)
        Z = q.icdf(Z)
    elif fam_z == 2:
        Z = torch.from_numpy(t.ppf(Z.numpy(), df=par2))
    elif fam_z == 3:
        q = Exponential(rate=1)
        Z = q.icdf(Z)
    else:
        raise Exception("fam_z must be 1, 2 or 3")

    beta_xz = torch.tensor(beta_xz).float()
    if beta_xz.dim()<2:
        beta_xz = beta_xz.unsqueeze(-1)
    _x_mu = torch.cat([torch.ones(*(X.shape[0],1)),Z],dim=1) @ beta_xz #XZ dependence (n x (1+d)) matmul (1+d x 1)
    # Look at X[:,0] - _x_mu[:,0]. Run KS test on that quantity for distribution with correct variance.
    # repeat for each "column". i.e. X[:,i] - _x_mu[:,i].
    # Think each of X and something to regress on Z. Think of the target distribtion. on what you expect post rejection sampling.

    if fam_x[1] == 4:
        mu = expit(_x_mu)
        d = Beta(concentration1=theta * mu, concentration0=theta * (1 - mu))
    elif fam_x[1] == 1:
        mu = _x_mu
        d = Normal(loc=mu, scale=theta) #ks -test uses this target distribution. KS-test on  0 centered d with scale phi...
        #might wanna consider d_X d's for more beta_XZ's
    elif fam_x[1] == 3:  # Change
        mu = torch.exp(_x_mu)
        d = Gamma(rate=1 / (mu * theta), concentration=1 / theta)
    else:
        raise Exception("fam_x must be 1, 3 or 4")
    wts = torch.zeros(*(X.shape[0],1))
    #To make Rejectoin sampling work, principal eigenvalue of target distribution i.e. d. Should be less than theta.
    p_z  = torch.zeros(*(X.shape[0],1))
    for i in range(d_X):
        _x = X[:,i].unsqueeze(-1)
        p_cond_z = d.log_prob(_x)
        p_z += p_cond_z
        _prob = p_cond_z- qden(_x)
        wts = wts + _prob
    wts = wts.exp()
    p_z = p_z.exp()
    if fam_x[0] == 1 and fam_x[1] == 1:
        max_ratio_points = mu * theta * phi / (theta * phi - theta)
        normalization = (d_X*d.log_prob(max_ratio_points) - d_X*qden(max_ratio_points)).exp()
    else:
        print("Warning: No analytical solution exist for maximum density ratio using defautl sample max")
        normalization = wts.max()
    if torch.isnan(wts).all():
        raise Exception("Problem with weights")
    wts_tmp = wts / normalization
    keep_index = (torch.rand_like(wts) < wts_tmp).squeeze()
    inv_wts = 1. / p_z #Variance of the weights seems to blow up when n is large, this also causes problems for the estimator...
    X,Y,Z,inv_wts = X[keep_index,:],Y[keep_index,:],Z[keep_index,:], inv_wts[keep_index]
    return X,Y,Z,inv_wts

def simulate_xyz_multivariate(n, oversamp,d_Z,beta_xy,beta_xz,yz,seed,d_Y=1,d_X=1,phi=2,theta=2,q_fac=1.0):
    """
    beta_xz has dim (d_Z+1) list
    beta_xy has dim 2 list
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    X,Y,Z,w = sim_multivariate_XYZ(oversamp, d_Z, n, beta_xy, beta_xz, yz, seed, par2=1, fam_z=1, fam_x=[1,1], phi=phi,theta=theta,d_X=d_X,d_Y=d_Y,q_fac=q_fac)
    while X.shape[0]<n:
        print(f'Undersampled: {X.shape[0]}')
        oversamp = oversamp*1.01
        X_new,Y_new,Z_new, w_new= sim_multivariate_XYZ(oversamp, d_Z, n, beta_xy, beta_xz, yz, seed, par2=1, fam_z=1, fam_x=[1,1], phi=phi,theta=theta,d_X=d_X,d_Y=d_Y,q_fac=q_fac)
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




