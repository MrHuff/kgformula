from sklearn import metrics
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from math import log
from kgformula.kernels import *

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


class input_block(torch.nn.Module):
    def __init__(self, input, output):
        super(input_block, self).__init__()

        self.f = nn.Sequential(nn.Linear(input, output),
                               torch.nn.Tanh(),
                               )
    def forward(self, x):
        return self.f(x)

class output_block(torch.nn.Module):
    def __init__(self, input, output):
        super(output_block, self).__init__()

        self.f = nn.Sequential(nn.Linear(input, output)
                               )
    def forward(self, x):
        return self.f(x)

class _res_block(torch.nn.Module):
    def __init__(self,input,output):
        super(_res_block, self).__init__()

        self.f = nn.Sequential(nn.Linear(input,output),
                                torch.nn.Tanh(),
                               )
    def forward(self,x):
        return self.f(x)+x


class MLP(torch.nn.Module):
    def __init__(self,d,f=12,k=2,o=1):
        super(MLP, self).__init__()
        self.model = nn.ModuleList()
        self.model.append(input_block(d, f))
        for i in range(k):
            self.model.append(_res_block(f, f))
        self.model.append(output_block(f, o))
    def pass_through(self,x):
        # X,Z = x.unbind(dim=1)
        for l in self.model:
            x = l(x)
        return x #+ (X.unsqueeze(-1)**2+Z.unsqueeze(-1)**2)

    def forward(self,input):
        output = []
        for i in range(1):
            neg = self.pass_through(input[i])
            pos = self.pass_through(input[i+1])
            output.append([neg,pos])
        return output

    def forward_val(self,input):
        pred = self.pass_through(input)
        return pred

    def get_w(self, x,z,empty=[]):
        return torch.exp(-self.forward_val(torch.cat([x,z],dim=1)) )

class TRE_net(torch.nn.Module):
    def __init__(self, dim, o, f, k, m):
        super(TRE_net, self).__init__()
        self.module_list = torch.nn.ModuleList()
        self.m = m
        for i in range(m):
            self.module_list.append(MLP(dim,f,k,o))

    def forward(self,input):
        output = []
        for i in range(self.m):
            neg = self.module_list[i].pass_through(input[i])
            pos = self.module_list[i].pass_through(input[i+1])
            output.append([neg,pos])
        return output

    def forward_val(self,input):
        pred = 0
        for i in range(self.m):
            pred += self.module_list[i].pass_through(input)
        return pred

    def get_w(self, x, z, empty=[]):
        dat = torch.cat([x,z],dim=1)
        base = 0
        for i in range(0,self.m):
            base += self.module_list[i].forward_val(dat)
        return torch.exp(-base)

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
        self.a_m = [] #m'=m-1, m. m'=2 -> m=3 ->
        self.a_0 = []
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
        self.mode='train'
    def divide_data(self):
        X_dat = torch.chunk(self.X,2,dim=0)
        self.X_joint = X_dat[0]
        self.X_pom = X_dat[1]
        Z_dat = torch.chunk(self.Z,2,dim=0)
        self.Z_joint = Z_dat[0]
        self.Z_pom = Z_dat[1]

    def set_mode(self,mode):
        self.mode=mode
        if self.mode=='train':
            self.X = self.X_train
            self.Z = self.Z_train
        elif self.mode=='val':
            self.X = self.X_val
            self.Z = self.Z_val
        self.divide_data()
        self.bs = int(round(self.bs_perc * self.X_joint.shape[0]))


class classification_dataset_Q(classification_dataset):
    def __init__(self,X,Z,X_q,bs=1.0,kappa=1,val_rate = 0.01):
        super(classification_dataset_Q, self).__init__(X,Z,bs,kappa,val_rate)
        self.X_q_train = X_q[~self.mask, :]
        self.X_q_val = X_q[self.mask,:]

    def divide_data(self):
        X_dat = torch.chunk(self.X,2,dim=0)
        self.X_joint = X_dat[0]
        Z_dat = torch.chunk(self.Z,2,dim=0)
        self.Z_joint = Z_dat[0]
        self.Z_pom = Z_dat[1]
        X_q_dat  = torch.chunk(self.X_q,2,dim=0)
        self.X_pom = X_q_dat[1]

    def set_mode(self, mode):
        self.mode = mode
        if self.mode == 'train':
            self.X = self.X_train
            self.Z = self.Z_train
            self.X_q = self.X_q_train

        elif self.mode == 'val':
            self.X = self.X_val
            self.Z = self.Z_val
            self.X_q = self.X_q_val

        self.divide_data()
        self.bs = int(round(self.bs_perc * self.X_joint.shape[0]))


# class classification_dataset_Q_TRE(classification_dataset_Q):
#     def __init__(self,X,Z,X_q,bs=1.0,kappa=1,val_rate = 0.01):
#         super(classification_dataset_Q_TRE, self).__init__(X,Z,X_q,bs,kappa,val_rate)
#
#     def get_sample(self):
#         i_s = np.random.randint(0, self.X_joint.shape[0] - self.bs-1)
#         joint_samp = torch.cat([self.X_joint[i_s:(i_s+self.bs),:],self.Z_joint[i_s:(i_s+self.bs),:]],dim=1)
#         i_s_2 = np.random.randint(0,  self.X_q_pom.shape[0]- self.bs*self.kappa-1)
#         X_p_samp  =self.X_pom[i_s_2:(i_s_2+self.bs*self.kappa),:]
#         pom_samp = torch.cat([X_p_samp, self.Z_pom[torch.randperm(self.Z_pom.shape[0])[:(self.bs*self.kappa)],:]],dim=1)
#         X_q_samp = self.X_q_pom[i_s:(i_s+self.bs),:]
#         return joint_samp,pom_samp,X_p_samp,X_q_samp
#
#     def get_val_sample(self):
#         n = min(self.X_joint.shape[0],self.X_pom.shape[0])
#         joint_samp = torch.cat([self.X_joint[:n,:],self.Z_joint[:n,:]],dim=1)
#         X_p_samp = self.X_pom[:n,:]
#         pom_samp = torch.cat([X_p_samp, self.Z_pom[torch.arange(self.Z_pom.shape[0]-1,-1,-1),:]],dim=1)
#         X_q_samp = self.X_q_pom[:n,:]
#         return joint_samp,pom_samp,n,X_p_samp,X_q_samp

class dataset_MI_TRE(Dataset):
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
        self.mode='train'

    def divide_data(self):
        X_dat = torch.chunk(self.X,2,dim=0)
        self.X_joint = X_dat[0]
        self.X_pom = X_dat[1]
        Z_dat = torch.chunk(self.Z,2,dim=0)
        self.Z_joint = Z_dat[0]
        self.Z_pom = Z_dat[1]


    def set_mode(self, mode):
        self.mode = mode
        if self.mode == 'train':
            self.X = self.X_train
            self.Z = self.Z_train

        elif self.mode == 'val':
            self.X = self.X_val
            self.Z = self.Z_val

        self.divide_data()
        self.bs = int(round(self.bs_perc * self.X_joint.shape[0]))

    # def get_sample(self):
    #     i_s = np.random.randint(0, self.X_joint.shape[0] - self.bs - 1)
    #     X_joint_samp,Z_joint_samp = self.X_joint[i_s:(i_s + self.bs), :], self.Z_joint[i_s:(i_s + self.bs), :]
    #     i_s_2 = np.random.randint(0, self.Z_pom.shape[0] - self.bs * self.kappa - 1)
    #     X_pom_samp,Z_pom_samp = self.X_pom[torch.randperm(self.X_pom.shape[0])[:(self.bs * self.kappa)], :],self.Z_pom[i_s_2:(i_s_2 + self.bs * self.kappa), :]
    #     data_out = [torch.cat([X_pom_samp,Z_pom_samp],dim=1)]
    #     for a_0, a_m in zip(self.a_0, self.a_m):
    #         transition_x = a_0 * X_pom_samp + a_m * X_joint_samp
    #         data_out.append(torch.cat([transition_x,Z_pom_samp],dim=1))
    #     data_out.append(torch.cat([X_joint_samp,Z_joint_samp],dim=1))
    #     return data_out

    # def get_val_sample(self):
    #     data_out = [torch.cat([self.X_pom, self.Z_pom], dim=1)]
    #     data_out.append(torch.cat([ self.X_joint, self.Z_joint],dim=1))
    #     return data_out

class dataset_rulsif(Dataset):
    def __init__(self,X,X_q,Z):
        self.x_joint,self.x_pom = torch.chunk(X,2)
        self.z_joint,self.z_pom = torch.chunk(Z,2)
        self.x_q_joint,self.x_q_pom = torch.chunk(X_q,2)
        self.joint = torch.cat([self.x_joint,self.z_joint],dim=1)
        self.pom_xz = torch.cat([self.x_pom[torch.randperm(self.x_pom.shape[0]),:],self.z_pom],dim=1)
        self.pom_x_q_z = torch.cat([self.x_q_pom[torch.randperm(self.x_q_pom.shape[0]),:],self.z_pom],dim=1)

    def get_data(self):
        return self.pom_xz,self.joint,self.pom_x_q_z

class rulsif(torch.nn.Module):
    def __init__(self,joint,pom,lambda_reg=1e-3,alpha=0.1):
        super(rulsif, self).__init__()
        self.joint  = joint
        self.alpha=alpha
        self.centers = pom
        self.nx = joint.shape[0]
        self.ny = pom.shape[0]
        ls = torch.median(torch.cdist(pom,pom))
        self.ker = RBFKernel()
        self.ker.lengthscale = ls
        self.register_buffer('diag',torch.eye(self.nx)*lambda_reg)

    def calc_theta(self):
        with torch.no_grad():
            phi_x = self.ker(self.centers,self.centers).evaluate()
            phi_y = self.ker(self.joint,self.centers).evaluate()
            H = self.alpha * (phi_x.t()@phi_x / self.nx) + (1 - self.alpha) * (phi_y.t() @phi_y / self.ny)
            h = phi_x.mean(dim=0,keepdim=True).t()
            self.theta,_ = torch.solve(h,H+self.diag)
            self.theta[self.theta<0]=0

    def get_w(self,X, Z,X_q_test):
        data = torch.cat([X,Z],dim=1)
        with torch.no_grad():
            alpha_density_ratio = self.ker(data, self.centers)@self.theta
        return alpha_density_ratio.clamp_min_(1e-30)



class dataset_MI_TRE_Q(Dataset):
    def __init__(self,X,X_q,Z,m,p=1,bs=1.0,val_rate = 0.01):
        self.m = m
        self.a_m = [(k/m)**p for k in range(1,m)] #m'=m-1, m. m'=2 -> m=3 ->
        self.a_0 = [(1-el**2)**0.5 for el in self.a_m]
        self.n=X.shape[0]
        mask = np.array([False] * self.n)
        mask[0:round(val_rate*self.n)] = True
        np.random.shuffle(mask)
        self.X_train = X[~mask,:]
        self.X_q_train = X_q[~mask,:]
        self.Z_train = Z[~mask,:]
        self.X_val = X[mask]
        self.X_q_val = X_q[mask]
        self.Z_val = Z[mask]
        self.bs_perc = bs
        self.device = X.device
        self.kappa = 1

    def divide_data(self):
        X_q_dat = torch.chunk(self.X_q,2,dim=0)
        self.X_pom = X_q_dat[1]
        X_dat = torch.chunk(self.X,2,dim=0)
        self.X_joint = X_dat[0]
        Z_dat = torch.chunk(self.Z,2,dim=0)
        self.Z_joint = Z_dat[0]
        self.Z_pom = Z_dat[1]

    def set_mode(self, mode):
        self.mode = mode
        if self.mode == 'train':
            self.X = self.X_train
            self.Z = self.Z_train
            self.X_q = self.X_q_train

        elif self.mode == 'val':
            self.X = self.X_val
            self.Z = self.Z_val
            self.X_q = self.X_q_val

        self.divide_data()
        self.bs = int(round(self.bs_perc * self.X_joint.shape[0]))

    # def train_mode(self):
    #     self.X = self.X_train
    #     self.X_q = self.X_q_train
    #     self.Z = self.Z_train
    #     self.divide_data()
    #     self.bs = int(round(self.bs_perc*self.X_joint.shape[0]))
    #
    # def val_mode(self):
    #     self.X = self.X_val
    #     self.X_q = self.X_q_val
    #     self.Z = self.Z_val
    #     self.divide_data()

    # def get_sample(self):
    #     i_s = np.random.randint(0, self.X_joint.shape[0] - self.bs - 1)
    #     X_joint_samp,Z_joint_samp = self.X_joint[i_s:(i_s + self.bs), :], self.Z_joint[i_s:(i_s + self.bs), :]
    #     i_s_2 = np.random.randint(0, self.X_pom.shape[0] - self.bs * self.kappa - 1)
    #     X_pom_samp,Z_pom_samp = self.X_pom[torch.randperm(self.X_pom.shape[0])[:(self.bs * self.kappa)], :],self.Z_pom[i_s_2:(i_s_2 + self.bs * self.kappa), :]
    #     data_out = [torch.cat([X_pom_samp,Z_pom_samp],dim=1)]
    #     for a_0, a_m in zip(self.a_0, self.a_m):
    #         transition_x = a_0 * X_pom_samp + a_m * X_joint_samp
    #         data_out.append(torch.cat([transition_x, Z_pom_samp], dim=1))
    #     data_out.append(torch.cat([X_joint_samp,Z_joint_samp],dim=1))
    #     return data_out
    #
    # def get_val_sample(self):
    #     data_out = [torch.cat([self.X_pom, self.Z_pom], dim=1)]
    #     data_out.append(torch.cat([ self.X_joint, self.Z_joint],dim=1))
    #     return data_out

class chunk_iterator(): #joint = pos, pom = neg
    def __init__(self,X_joint,Z_joint,X_pom,Z_pom,shuffle,batch_size,kappa=10,TRE=False,a_0=[],a_m=[],mode='train'):
        self.mode = mode
        self.X_joint = X_joint
        self.Z_joint = Z_joint
        self.X_pom = X_pom
        self.Z_pom = Z_pom
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n_joint = self.X_joint.shape[0]
        self.n_pom = self.X_pom.shape[0]
        self.chunks_joint = self.n_joint // batch_size + 1
        self.perm_joint = torch.randperm(self.n_joint)
        self.kappa=kappa
        if self.shuffle:
            self.X_joint = self.X_joint[self.perm_joint,:]
            self.Z_joint = self.Z_joint[self.perm_joint,:]

        if self.mode=='train':
            self.it_X = torch.chunk(self.X_joint,self.chunks_joint)
            self.it_Z = torch.chunk(self.Z_joint,self.chunks_joint)
        elif self.mode=='val':
            val_n = min(self.n_joint,self.n_pom)
            self.chunks_joint = val_n // batch_size + 1
            self.perm_pom = torch.randperm(self.n_pom)
            self.X_pom = self.X_pom[self.perm_pom,:]
            self.X_pom = self.X_pom[:val_n,:]
            self.Z_pom = self.Z_pom[:val_n,:]
            self.X_joint = self.X_joint[:val_n,:]
            self.Z_joint = self.Z_joint[:val_n,:]
            self.it_X = torch.chunk(self.X_joint, self.n_joint)
            self.it_Z = torch.chunk(self.Z_joint, self.n_joint)
            self.it_X_pom = torch.chunk(self.X_pom, self.n_joint)
            self.it_Z_pom = torch.chunk(self.Z_pom, self.n_joint)

        self.true_chunks = len(self.it_X)
        self._index = 0
        self.TRE = TRE
        self.a_0=a_0
        self.a_m=a_m

    def __next__(self):
        ''''Returns the next value from team object's lists '''
        if self.mode=='train':
            if self._index < self.true_chunks:
                a,b = self.it_X[self._index],self.it_Z[self._index]
                n_ = a.shape[0]*self.kappa
                pom_perm = torch.randperm(self.n_pom)[:n_]
                c,d = self.X_pom[pom_perm,:],self.Z_pom[:n_]
                data_out = [torch.cat([c, d], dim=1)]
                if self.TRE:
                    for a_0, a_m in zip(self.a_0, self.a_m):
                        transition_x = a_0 * c + a_m * a
                        data_out.append(torch.cat([transition_x, d], dim=1))
                data_out.append(torch.cat([a, b], dim=1))
                self._index += 1
                return data_out
            raise StopIteration

        else:
            if self._index < self.true_chunks:
                a,b = self.it_X[self._index],self.it_Z[self._index]
                c,d= self.it_X_pom[self._index],self.it_Z_pom[self._index]
                data_out = [torch.cat([c, d], dim=1)]
                data_out.append(torch.cat([a, b], dim=1))
                self._index += 1
                return data_out
            raise StopIteration

    def __len__(self):
        return self.true_chunks

class NCE_dataloader():
    def __init__(self,dataset,bs_ratio,shuffle=False,kappa=10,TRE=False):
        self.dataset = dataset
        self.dataset.set_mode('train')
        self.bs_ratio = bs_ratio
        self.batch_size = int(round(self.dataset.X.shape[0] * bs_ratio))
        self.shuffle = shuffle
        self.n = self.dataset.X_joint.shape[0]
        self.len=self.n//self.batch_size+1
        self.kappa=kappa
        self.TRE=TRE
    def __iter__(self):
        if self.dataset.mode=='train':
            self.batch_size = int(round(self.dataset.X_joint.shape[0] * self.bs_ratio))
        else:
            self.batch_size = self.dataset.X_joint.shape[0]//5
        return chunk_iterator(
                                X_joint=self.dataset.X_joint,
                                Z_joint=self.dataset.Z_joint,
                                X_pom=self.dataset.X_pom,
                                Z_pom=self.dataset.Z_pom,
                                kappa=self.kappa,
                                a_m=self.dataset.a_m,
                                a_0=self.dataset.a_0,
                                TRE=self.TRE,
                                mode=self.dataset.mode,
                              shuffle=self.shuffle,
                              batch_size=self.batch_size)

    def __len__(self):
        if self.dataset.mode=='train':
            self.batch_size = int(round(self.dataset.X_joint.shape[0] * self.bs_ratio))
        else:
            self.batch_size = self.dataset.X_joint.shape[0]//5
        return  len(chunk_iterator(
                                X_joint=self.dataset.X_joint,
                                Z_joint=self.dataset.Z_joint,
                                X_pom=self.dataset.X_pom,
                                Z_pom=self.dataset.Z_pom,
                                kappa=self.kappa,
                                a_m=self.a_m,
                                a_0=self.a_0,
                                TRE=self.TRE,
                                mode=self.dataset.mode,
                              shuffle=self.shuffle,
                              batch_size=self.batch_size))
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
    def forward(self,fake_preds,true_preds):
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
