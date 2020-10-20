import torch
from sklearn import metrics
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from math import log
from torch.distributions import *

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class CustomSwish(nn.Module):
    def forward(self, input_tensor):
        return Swish.apply(input_tensor)
class _res_block(torch.nn.Module):
    def __init__(self,input,output):
        super(_res_block, self).__init__()

        self.f = nn.Sequential(nn.Linear(input,output),
                               CustomSwish(),
                               )
    def forward(self,x):
        return self.f(x)

class MLP_shared(torch.nn.Module): #try new architecture...
    def __init__(self,input_dim,latent_size,depth_main,outputs,depth_task):
        super(MLP_shared, self).__init__()
        self.main = MLP(input_dim,latent_size,depth_main,latent_size)
        self.tasks = nn.ModuleList()
        for d in outputs:
            self.tasks.append(MLP(latent_size,latent_size,depth_task,d))

    def forward(self,x,bridge_indicator):
        l =self.main(x)
        output = []
        for i,m in enumerate(self.tasks):
            s = l[(bridge_indicator==i) | (bridge_indicator==i+1) ,:]
            output.append(m(s))
        return output

    def predict(self,x):
        l = self.main(x)
        output = []
        for i, m in enumerate(self.tasks):
            output.append(m(l))
        return output


class MLP_pq(torch.nn.Module):
    def __init__(self,d_p,d_q,f=12,k=2,o=1):
        super(MLP_pq, self).__init__()
        self.model_p = MLP(d_p,f,k,o)
        self.model_q = MLP(d_q,f,k,o)

    def forward(self, input_p, input_q):
        return self.model_p(input_p), self.model_q(input_q)

    def forward_p(self, input_p):
        return self.model_p(input_p)

    def forward_q(self, input_q):
        return self.model_q(input_q)

    def get_w(self,x_p,z_p,x_q):
        p,q = self.forward(torch.cat([x_p,z_p],dim=1),torch.cat([x_q,x_p],dim=1))
        return torch.exp(-(p+q))

class TRE(torch.nn.Module):
    def __init__(self,input_dim_u,u_out_dim,width,depth_u,input_dim_v,v_out_dims,depth_v,IP=True):
        super(TRE, self).__init__()
        self.g_u = MLP(d=input_dim_u,f=width,k=depth_u,o=u_out_dim)
        self.f_k = MLP_shared(input_dim=input_dim_v,latent_size=width,depth_main=depth_v,outputs=[v*u_out_dim for v in v_out_dims],depth_task=depth_v)
        self.IP = IP
        if not self.IP:
            self.W = nn.ParameterList([nn.Parameter(torch.randn(u_out_dim, u_out_dim),requires_grad=True) for i in range(len(v_out_dims))])
    def forward(self,u,v,indicator):
        g_u = self.g_u(u).repeat(2,1) #bsxdim
        list_of_fk = self.f_k(v,indicator) #[bsxdimxv_out_dims]
        #1. Try IP

        if self.IP:
            return (g_u.unsqueeze(-1)*torch.stack(list_of_fk,dim=-1)).sum(dim=1).squeeze()
        else:
            output = [ torch.bmm((g_u@w).unsqueeze(1),fk.unsqueeze(-1)) for w,fk in zip(self.W,list_of_fk)] #bs x dim
            return torch.stack(output,dim=-1).squeeze()

    def predict(self,x,z):
        g_u = self.g_u(x) #bsxdim
        list_of_fk = self.f_k.predict(z)
        if self.IP:
            return (g_u.unsqueeze(-1) * torch.stack(list_of_fk, dim=-1)).sum(dim=1).squeeze().sum(dim=-1)
        else:
            output = [torch.bmm((g_u @ w).unsqueeze(1), fk.unsqueeze(-1)) for w, fk in
                      zip(self.W, list_of_fk)]  # bs x dim
            return torch.stack(output, dim=-1).squeeze().sum(dim=-1)

    def get_w(self,x,z):
        return torch.exp(self.predict(x,z))

class MLP(torch.nn.Module):
    def __init__(self,d,f=12,k=2,o=1):
        super(MLP, self).__init__()
        self.model = nn.ModuleList()
        self.model.append(_res_block(d, f))
        for i in range(k):
            self.model.append(_res_block(f, f))
        self.model.append(_res_block(f, o))

    def forward(self,x):
        # X,Z = x.unbind(dim=1)
        for l in self.model:
            x = l(x)
        return x #+ (X.unsqueeze(-1)**2+Z.unsqueeze(-1)**2)

    def get_w(self, x,z,empty=[]):
        return torch.exp(-self.forward(torch.cat([x,z],dim=1)))

class logistic_regression(torch.nn.Module):
    def __init__(self,d):
        super(logistic_regression, self).__init__()
        self.W = torch.nn.Linear(in_features=d,  out_features=1,bias=True)

    def forward(self,X):
        return self.W(X)

    def get_w(self, x,z):
        return torch.exp(self.forward(torch.cat([x,z],dim=1)))

class classification_dataset(Dataset):
    def __init__(self,X,Z,bs=1.0,kappa=1,val_rate = 0.01):
        super(classification_dataset, self).__init__()
        self.n=X.shape[0]
        self.mask = np.array([False] * self.n)
        self.mask[0:round(val_rate*self.n)] = True
        np.random.shuffle(self.mask)
        self.X_train = X[~self.mask,:]
        self.Z_train = Z[~self.mask,:]
        self.X_val = X[self.mask]
        self.Z_val = Z[self.mask]
        self.bs_perc = bs
        self.device = X.device
        self.kappa = kappa
        self.train_mode()

    def divide_data(self):
        X_dat = torch.chunk(self.X,2,dim=0)
        self.X_joint = X_dat[0]
        self.X_pom = X_dat[1]
        Z_dat = torch.chunk(self.Z,2,dim=0)
        self.Z_joint = Z_dat[0]
        self.Z_pom = Z_dat[1]

    def train_mode(self):
        self.X = self.X_train
        self.Z = self.Z_train
        self.divide_data()
        self.bs = int(round(self.bs_perc*self.X_joint.shape[0]))

    def val_mode(self):
        self.X = self.X_val
        self.Z = self.Z_val
        self.divide_data()

    def get_sample(self):
        i_s = np.random.randint(0, self.X_joint.shape[0] - self.bs-1)
        joint_samp = torch.cat([self.X_joint[i_s:(i_s+self.bs),:],self.Z_joint[i_s:(i_s+self.bs),:]],dim=1)
        i_s_2 = np.random.randint(0, self.X_pom.shape[0] - self.bs*self.kappa-1)
        pom_samp = torch.cat([self.X_pom[i_s_2:(i_s_2+self.bs*self.kappa),:], self.Z_pom[torch.randperm(self.X_pom.shape[0])[:(self.bs*self.kappa)],:]],dim=1)
        return joint_samp,pom_samp

    def get_val_sample(self):
        n = min(self.X_joint.shape[0],self.X_pom.shape[0])
        joint_samp = torch.cat([self.X_joint[:n,:],self.Z_joint[:n,:]],dim=1)
        pom_samp = torch.cat([self.X_pom[:n,:], self.Z_pom[torch.arange(self.Z_pom.shape[0],0,-1),:]],dim=1)
        return joint_samp,pom_samp,n


class classification_dataset_Q(classification_dataset):
    def __init__(self,X,Z,X_q,bs=1.0,kappa=1,val_rate = 0.01):
        super(classification_dataset_Q, self).__init__(X,Z,bs,kappa,val_rate)
        self.X_q_train = X_q[~self.mask, :]
        self.X_q_val = X_q[self.mask,:]
        self.train_mode()

    def divide_data(self):
        X_dat = torch.chunk(self.X,2,dim=0)
        self.X_joint = X_dat[0]
        self.X_pom = X_dat[1]
        Z_dat = torch.chunk(self.Z,2,dim=0)
        self.Z_joint = Z_dat[0]
        self.Z_pom = Z_dat[1]
        X_q_dat  = torch.chunk(self.X_q,2,dim=0)
        self.X_q_pom = X_q_dat[1]

    def train_mode(self):
        self.X = self.X_train
        self.Z = self.Z_train
        self.X_q = self.X_q_train
        self.divide_data()
        self.bs = int(round(self.bs_perc*self.X_joint.shape[0]))

    def val_mode(self):
        self.X = self.X_val
        self.Z = self.Z_val
        self.X_q = self.X_q_val
        self.divide_data()

    def get_sample(self):
        i_s = np.random.randint(0, self.X_joint.shape[0] - self.bs-1)
        joint_samp = torch.cat([self.X_joint[i_s:(i_s+self.bs),:],self.Z_joint[i_s:(i_s+self.bs),:]],dim=1)
        i_s_2 = np.random.randint(0, self.X_q_pom.shape[0] - self.bs*self.kappa-1)
        pom_samp = torch.cat([self.X_q_pom[i_s_2:(i_s_2+self.bs*self.kappa),:], self.Z_pom[torch.randperm(self.X_pom.shape[0])[:(self.bs*self.kappa)],:]],dim=1)
        return joint_samp,pom_samp

    def get_val_sample(self):
        n = min(self.X_joint.shape[0],self.X_q_pom.shape[0])
        joint_samp = torch.cat([self.X_joint[:n,:],self.Z_joint[:n,:]],dim=1)
        pom_samp = torch.cat([self.X_q_pom[:n,:], self.Z_pom[torch.arange(self.Z_pom.shape[0],0,-1),:]],dim=1)
        return joint_samp,pom_samp,n

class classification_dataset_Q_TRE(classification_dataset_Q):
    def __init__(self,X,Z,X_q,bs=1.0,kappa=1,val_rate = 0.01):
        super(classification_dataset_Q_TRE, self).__init__(X,Z,X_q,bs,kappa,val_rate)

    def get_sample(self):
        i_s = np.random.randint(0, self.X_joint.shape[0] - self.bs-1)
        joint_samp = torch.cat([self.X_joint[i_s:(i_s+self.bs),:],self.Z_joint[i_s:(i_s+self.bs),:]],dim=1)
        i_s_2 = np.random.randint(0, self.X_pom.shape[0] - self.bs*self.kappa-1)
        X_p_samp  =self.X_pom[i_s_2:(i_s_2+self.bs*self.kappa),:]
        pom_samp = torch.cat([X_p_samp, self.Z_pom[torch.randperm(self.Z_pom.shape[0])[:(self.bs*self.kappa)],:]],dim=1)
        X_q_samp = self.X_q_pom[self.X_q_pom[i_s:(i_s+self.bs),:],:]
        return joint_samp,pom_samp,X_p_samp,X_q_samp

    def get_val_sample(self):
        n = min(self.X_joint.shape[0],self.X_pom.shape[0])
        joint_samp = torch.cat([self.X_joint[:n,:],self.Z_joint[:n,:]],dim=1)
        X_p_samp = self.X_pom[:n,:]
        pom_samp = torch.cat([X_p_samp, self.Z_pom[torch.arange(self.Z_pom.shape[0],0,-1),:]],dim=1)
        X_q_samp = self.X_q_pom[:n,:]

        return joint_samp,pom_samp,n,X_p_samp,X_q_samp


class Log1PlusExp(torch.autograd.Function):
    """Implementation of x ↦ log(1 + exp(x))."""
    @staticmethod
    def forward(ctx, x):
        exp = x.exp_()
        ctx.save_for_backward(x)
        return x.where(torch.isinf(exp), exp.log1p_())
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output / (1 + (-x).exp())

log_1_plus_exp = Log1PlusExp.apply


class NCE_objective_stable(torch.nn.Module):
    def __init__(self,kappa=1.):
        super(NCE_objective_stable, self).__init__()
        self.kappa = kappa
        self.log_kappa = log(kappa)
    def forward(self,true_preds,fake_preds):
        _err = torch.log(1+torch.exp(-true_preds+self.log_kappa))+torch.log(1+torch.exp(fake_preds-self.log_kappa)).sum(dim=1,keepdim=True)
        return _err.sum()

class standard_bce(torch.nn.Module):
    def __init__(self,pos_weight =1.0):
        super(standard_bce, self).__init__()
        self.obj = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    def forward(self,pred,true):
        return self.obj(pred,true)

def accuracy_check(y_pred,Y):
    with torch.no_grad():
        y_pred = (y_pred.float() > 0.5).cpu().float().numpy()
        return np.mean(Y.cpu().numpy()==y_pred)

def auc_check(y_pred,Y):
    with torch.no_grad():
        y_pred = (y_pred.float() > 0.5).cpu().float().numpy()
        fpr, tpr, thresholds = metrics.roc_curve(Y.cpu().numpy(), y_pred, pos_label=1)
        auc =  metrics.auc(fpr, tpr)
        return auc
