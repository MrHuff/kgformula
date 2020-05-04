import torch
from torch.distributions import Normal,Beta,Gamma,Exponential
from pycopula.copula import ArchimedeanCopula
from pycopula.simulation import simulate
from scipy.stats import t,gamma
import numpy as np
import warnings
import math

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

def sim_X(n,dist,theta):
    if dist==1:
        d = Normal(loc=0,scale=theta)
    elif dist== 4:
        d = Beta(concentration0=theta,concentration1=theta)
    elif dist== 3:
        d = Gamma(concentration=1,rate=1/theta) #Exponential theta
    else:
        raise Exception("X distribution must be normal (1), beta (4) or gamma (3)")
    return {'data':d.sample((n,1)),'density':d.log_prob}


def rnormCopula(n=100,mean = torch.zeros(*(2,1)),cov=torch.eye(2)):
    return torch.f

def rnormCopula2(n=100,mean = torch.zeros(*(2,1)),cov=torch.eye(2),df=1):
    if cov.shape == torch.Size([2,2]):
        l = torch.cholesky(cov)
        M = mean.t() #nx2
        M = M.repeat(n,1)
        samples = M + torch.randn(*(n,2))@l
    else:
        M = mean.t()
        M = M.repeat(n,1)
        v = torch.randn(n)
        samples = M + torch.stack([v,v*cov[:,0]+torch.randn(n)*cov[:,1]],dim=1)
    return torch.from_numpy(t.cdf(samples.numpy(),df=df)).float()

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

def sim_XYZ(n, beta, cor, phi=1, theta=1, par2=1,fam=1, fam_x=[1,1], fam_y=1, fam_z=1,oversamp = 10, seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if oversamp<1:
        warnings.warn("Oversampling rate must be at least 1... changing")
        oversamp=1

    if type(cor) is not list: #cor controls x xz relation!
        cor = torch.tensor([cor,0]).unsqueeze(-1)
    else:
        cor = torch.tensor(cor).unsqueeze(-1)

    N = round(oversamp*n)
    tmp = sim_X(N,fam_x[0],theta)
    dat = tmp['data']
    qden = tmp['density']
  ## add in extra columns for Y and Zs
  ## get Copula value
    dat = sim_UV(dat, fam, cor, par2) #ZY depedence
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
        d = Beta(concentration1=phi*mu,concentration0=phi*(1-mu))
        _prob = d.log_prob(dat[:,0])-qden(dat[:,0])
        wts = _prob.exp()
    elif fam_x[1]==1:
        mu = X
        d = Normal(loc = mu,scale = phi**0.5)
        _prob = d.log_prob(dat[:,0])-qden(dat[:,0])
        wts = _prob.exp()
    elif fam_x[1]==3: #Change
        mu = torch.exp(X)
        d = Gamma(rate=1/(mu*phi),concentration=1/phi)
        _prob = d.log_prob(dat[:,0])-qden(dat[:,0])
        wts = _prob.exp()
    else:
        raise Exception("fam_x must be 1, 3 or 4")
  ## reject samples based on value of X|Z
    if torch.isnan(wts).all():
        raise Exception("Problem with weights")
    inv_wts=1/wts
    wts = wts/wts.max()
    keep_index = torch.rand_like(wts)<wts
    dat = dat[keep_index,:]
    return dat,inv_wts[keep_index]

def simulate_xyz(n, beta, cor, phi=1, theta=1, par2=1,fam=1, fam_x=[1,1], fam_y=1, fam_z=1,oversamp = 10, seed=1):
    data,w = sim_XYZ(n, beta, cor, phi,theta, par2,fam, fam_x, fam_y, fam_z,oversamp, seed)
    if data.shape[0]<n:
        print(f'Undersampled: {data.shape[0]}')
        oversamp = n/data.shape[0]*1.5
        data,w = sim_XYZ(n, beta, cor, phi, theta, par2, fam, fam_x, fam_y, fam_z, oversamp, seed)
    else:
        print(f'Ok: {data.shape[0]}')
        data = data[0:n,:]
    return data[:,0].unsqueeze(-1),data[:,1].unsqueeze(-1),data[:,2].unsqueeze(-1),w[0:n]

def sample_naive_multivariate(n,d_X,d_Z,d_Y,beta_xz,beta_xy,seed):
    torch.manual_seed(seed)
    Z = torch.randn(n,d_Z)
    X = beta_xz*Z[:,0:d_X]+(1+beta_xz)*torch.randn(n,d_X)
    Y = beta_xy*X[:,0:d_Y]**3+0.1*torch.randn(n,d_Y)+beta_xy/3*Z[:,0:d_Y]**3
    w = torch.ones(n,1)
    return X,Y,Z,w

def sim_multivariate_UV(dat,fam,par,par2,d_z):
    if not fam in [1,2,3,4,5,6,11]:
        raise Exception("family not supported")

    N = dat.shape[0]
    pars = torch.cat([torch.ones_like(dat),dat],dim=1)@par #Don't understand this part.  I think this is a Nx1 matrix?
    pars = pars.squeeze()

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
        sigma = torch.eye(d_z)
        sigma[torch.triu(torch.ones_like(sigma))==1] = pars
        sigma[torch.tril(torch.ones_like(sigma))==1] = pars

    elif fam in [2]:

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

def sim_multivariate_XYZ(oversamp,d_Z,n,xy,xz,yz):
    ref_dim = nCr(d_Z,2)
    if type(yz) is not list:  # cor controls x xz relation!
        cor = torch.tensor([yz, 0]).unsqueeze(-1)
    else:
        cor = torch.tensor(yz).unsqueeze(-1)
    cor = torch.cat([cor for i in range(ref_dim)],dim=1)




