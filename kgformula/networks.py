import torch
from sklearn import metrics
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import time
from math import log

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
        for l in self.model:
            x = l(x)
        return x

    def get_w(self, x,z):
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
        mask = np.array([False] * self.n)
        mask[0:round(val_rate*self.n)] = True
        np.random.shuffle(mask)
        self.X_train = X[~mask,:]
        self.Z_train = Z[~mask,:]
        self.X_val = X[mask]
        self.Z_val = Z[mask]
        self.bs_perc = bs
        self.device = X.device
        self.kappa = kappa
        self.train_mode()

    def train_mode(self):
        self.X = self.X_train
        self.Z = self.Z_train
        self.sample_indices_base = np.arange(self.X.shape[0])
        self.bs = int(round(self.bs_perc*self.X.shape[0]))

    def val_mode(self):
        self.X = self.X_val
        self.Z = self.Z_val
        self.sample_indices_base = np.arange(self.X.shape[0])
        self.bs = self.X.shape[0]

    def build_sampling_set(self,true_indices):
        np_cat = []
        for x in np.nditer(true_indices):
            np_cat.append(np.delete(self.sample_indices_base,x)[None,:])
        return np.concatenate(np_cat,axis=0)

    def sample_no_replace(self,fake_set,kappa,replace=False):
        np_cat = []
        for row in fake_set:
            np_cat.append(np.random.choice(row,kappa,replace))
        return np.concatenate(np_cat)

    def get_indices(self):
        if self.bs ==self.X.shape[0]:
            true_indices = self.sample_indices_base
        else:
            i_s = np.random.randint(0, self.X.shape[0] - 1 - self.bs)
            true_indices = np.arange(i_s,i_s+self.bs)
        fake_set = self.build_sampling_set(true_indices)
        fake_indices = self.sample_no_replace(fake_set,self.kappa,False)
        return true_indices,fake_indices#,HSIC_ref_indices_true,HSIC_ref_indices_fake

    def get_sample(self):
        T,F = self.get_indices()
        return torch.cat([self.X[T,:],self.Z[T,:]],dim=1),torch.cat([self.X[T.repeat(self.kappa),:],self.Z[F,:]],dim=1)

class classification_dataset_TRE(classification_dataset):
    def __init__(self,X,Z,m,p=1,bs=1.0,val_rate = 0.01):
        self.m = m
        self.a_m = [(k/m)**p for k in range(1,m)] #m'=m-1, m. m'=2 -> m=3 ->
        self.a_0 = [(1-el**2)**0.5 for el in self.a_m]
        self.n=X.shape[0]
        mask = np.array([False] * self.n)
        mask[0:round(val_rate*self.n)] = True
        np.random.shuffle(mask)
        self.X_train = X[~mask,:]
        self.Z_train = Z[~mask,:]
        self.X_val = X[mask]
        self.Z_val = Z[mask]
        self.bs_perc = bs
        self.device = X.device
        self.kappa = 1
        self.train_mode()
    def get_permute(self):
        T,F = self.get_indices()
        return self.X[T,:],self.Z[T,:],self.Z[F,:]

    def train_mode(self):
        self.X = self.X_train
        self.Z = self.Z_train
        self.sample_indices_base = np.arange(self.X.shape[0])
        self.bs = int(round(self.bs_perc*self.X.shape[0]))
        self.indicator = torch.tensor([self.bs*[k] for k in range(0,self.m+1)]).flatten() # m=2 [1,1] 0,1,2
        self.y = torch.tensor([True]*self.bs+[False]*self.bs)

    def val_mode(self):
        self.X = self.X_val
        self.Z = self.Z_val
        self.sample_indices_base = np.arange(self.X.shape[0])
        self.bs = self.X.shape[0]
        self.indicator = torch.tensor([self.bs*[k] for k in range(0,self.m+1)]).flatten()
        self.y = torch.tensor([True]*self.bs+[False]*self.bs)

    def get_sample(self): #Finish build the rest of the NN
        x,z_0,z_m = self.get_permute()
        data = [z_0]
        for a_0,a_m in zip(self.a_0,self.a_m):
            data.append(a_0*z_0+a_m*z_m)
        data.append(z_m)
        return x,torch.cat(data,dim=0),self.indicator,self.y

    def get_val_classification_sample(self):
        X = self.X.repeat(2,1)
        Z = torch.cat([self.Z,self.Z[torch.randperm(self.Z.shape[0])]])
        label = torch.tensor([True]*self.X.shape[0]+[False]*self.X.shape[0])
        return X,Z,label


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
