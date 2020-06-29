import torch
from sklearn import metrics
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import time

class _res_block(torch.nn.Module):
    def __init__(self,input,output):
        super(_res_block, self).__init__()

        self.f = nn.Sequential(nn.Linear(input,output),
                               nn.BatchNorm1d(output),
                               nn.Tanh(),
                               )
    def forward(self,x):
        return self.f(x)+x

class TDR(torch.nn.Module):
    def __init__(self,input_dim,latent_size,depth_main,outputs,depth_task):
        super(TDR, self).__init__()
        self.main = MLP(input_dim,latent_size,depth_main,latent_size)
        self.tasks = nn.ModuleList()
        for d in outputs:
            self.tasks.append(MLP(latent_size,latent_size,depth_task,d))

    def forward(self,X,Z):
        x = torch.cat([X,Z],dim=1)
        l =self.main(x)
        output = []
        for m in self.tasks:
            output.append(m(l))
        return output

class MLP(torch.nn.Module):
    def __init__(self,d,f=12,k=2,o=1):
        super(MLP, self).__init__()
        self.model = nn.ModuleList()
        self.model.append(nn.Linear(d,f))
        for i in range(k):
            self.model.append(_res_block(f, f))
        self.model.append(nn.Linear(f, o))

    def forward(self,X,Z):
        x = torch.cat([X,Z],dim=1)
        for l in self.model:
            x = l(x)
        return x

    def forward_predict(self,X,Z):
        return self.forward(X,Z)

    def get_w(self, x, y):
        return torch.exp(-self.forward(x,y))

class logistic_regression(torch.nn.Module):
    def __init__(self,d):
        super(logistic_regression, self).__init__()
        self.W = torch.nn.Linear(in_features=d,  out_features=1,bias=True)

    def forward(self,x,z):
        X = torch.cat([x,z],dim=1)
        return self.W(X)

    def forward_predict(self,X,Z):
        return self.forward(X,Z)

    def get_w(self, x, y):
        return torch.exp(-self.forward(x,y))

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
        return self.X[T,:],self.Z[T,:],self.X[T.repeat(self.kappa),:],self.Z[F,:]

def nu_sigmoid(x,kappa):
    return 1./(1+kappa*torch.exp(-x))

class Log1PlusExp(torch.autograd.Function):
    """Implementation of x ↦ log(1 + exp(x))."""
    @staticmethod
    def forward(ctx, x):
        exp = torch.exp(x)
        ctx.save_for_backward(x)
        return x.where(torch.isinf(exp), exp.log1p_())
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output / (1 + (-x).exp())

log_1_plus_exp = Log1PlusExp.apply

def NCE_objective(true_preds,fake_preds,kappa):
    _err = -torch.log(nu_sigmoid(true_preds,kappa)) - (1.-nu_sigmoid(fake_preds,kappa)).log().sum(dim=1)
    return _err.mean()

def NCE_objective_stable(true_preds,fake_preds,kappa=1):
    _err = -(true_preds-log_1_plus_exp(true_preds)-log_1_plus_exp(fake_preds).mean(dim=1))
    return _err.mean()

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
