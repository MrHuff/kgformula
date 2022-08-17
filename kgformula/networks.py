import torch.nn
from sklearn import metrics
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from math import log
from kgformula.kernels import *

def get_binary_mask(X):
    dim = X.shape[1]
    mask_ls = [0] * dim
    label_size = []
    for i in range(dim):
        x = X[:, i]
        un_el = x.unique()
        mask_ls[i] = un_el.numel() <= 10
        if mask_ls[i]:
            label_size.append(un_el.numel())
    return torch.tensor(mask_ls)

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

class nn_node(torch.nn.Module): #Add dropout layers, Do embedding layer as well!
    def __init__(self,d_in,d_out,cat_size_list,transformation=torch.tanh):
        super(nn_node, self).__init__()

        self.has_cat = len(cat_size_list)>0
        self.latent_col_list = []
        print('cat_size_list',cat_size_list)
        for i,el in enumerate(cat_size_list):
            translate_dict = { int(cat):j  for j,cat in enumerate(el)}
            nr_of_uniques = len(el)
            col_size = nr_of_uniques//2+4
            setattr(self,f'embedding_{i}',torch.nn.Embedding(nr_of_uniques,col_size))
            setattr(self,f'translate_dict_{i}',translate_dict)
            self.latent_col_list.append(col_size)
        self.lnorm = torch.nn.LayerNorm(d_out)
        self.w = torch.nn.Linear(d_in+sum(self.latent_col_list),d_out)
        self.f = transformation

    def forward(self,X,x_cat=[]):
        if not isinstance(x_cat,list):
            seq = torch.unbind(x_cat,1)
            cat_vals = [X]
            _dev = X.device
            for i,f in enumerate(seq):
                # print(i,f)
                # print(x_cat)
                f=f.cpu()
                ts_dict = getattr(self,f'translate_dict_{i}')
                f.apply_(lambda x: ts_dict[x])
                f = f.long().to(_dev)
                o = getattr(self,f'embedding_{i}')(f)
                cat_vals.append(o)
            X = torch.cat(cat_vals,dim=1)
        return self.f(self.lnorm(self.w(X)))

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
        return self.f(x) + x


class MLP(torch.nn.Module):
    def __init__(self,d,cat_marker,cat_size_list,f=12,k=2,o=1):
        super(MLP, self).__init__()
        self.cat_marker=cat_marker
        self.cat_size_list = cat_size_list
        self.model = nn.ModuleList()
        if k==0:
            self.first_layer = nn_node(d_in=d,d_out=1,cat_size_list=cat_size_list)
        else:
            self.first_layer = nn_node(d_in=d,d_out=f,cat_size_list=cat_size_list)
            for i in range(k-1):
                self.model.append(_res_block(f, f))
            self.model.append(output_block(f, o))
        print(self)

    def pass_through(self,x):
        # X,Z = x.unbind(dim=1)
        cont_x = x[:,~self.cat_marker]
        cat_x= x[:,self.cat_marker].long()
        x = self.first_layer(cont_x,cat_x)
        for l in self.model:
            x = l(x)
        return x #+ (X.unsqueeze(-1)**2+Z.unsqueeze(-1)**2)
    #TODO: Solution exists, just have to reproduce it and understand why it works!
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
        # return self.forward_val(torch.cat([x,z],dim=1))
        # return torch.exp(-self.forward_val(torch.cat([x,z],dim=1)) ) #SWITCHED! this will give you p(x|z)/q(x_q)
        return torch.exp(-self.forward_val(torch.cat([x,z],dim=1)) ) #SWITCHED! this will give you p(x|z)/q(x_q)
            # self.forward_val(torch.cat([x,z],dim=1))
            # torch.exp(-self.forward_val(torch.cat([x,z],dim=1)) ) #SWITCHED!

class TRE_net(torch.nn.Module):
    def __init__(self, dim, o, f, k, m,cat_marker,cat_size_list):
        super(TRE_net, self).__init__()
        self.module_list = torch.nn.ModuleList()
        self.cat_marker=cat_marker
        self.cat_size_list = cat_size_list

        self.m = m
        for i in range(m):
            self.module_list.append(MLP(dim,self.cat_marker,self.cat_size_list,f,k,o))
        print(self)

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
        return torch.exp(-self.forward(torch.cat([x,z],dim=1)))

class cat_dataset(Dataset):
    def __init__(self,X,Z,bs=1.0,val_rate=0.01):
        super(cat_dataset, self).__init__()
        self.n=X.shape[0]
        self.bs_perc = bs
        self.mask = np.array([False] * self.n)
        self.mask[0:round(val_rate*self.n)] = True
        np.random.shuffle(self.mask)
        self.X_train = X[~self.mask,:]
        self.Z_train = Z[~self.mask,:]
        self.X_val = X[self.mask]
        self.Z_val = Z[self.mask]

    def set_mode(self, mode):
        self.mode = mode
        if self.mode == 'train':
            self.X = self.X_train
            self.Z = self.Z_train
        elif self.mode == 'val':
            self.X = self.X_val
            self.Z = self.Z_val
        ref_bs = int(round(self.bs_perc * self.X.shape[0]))
        self.bs =ref_bs
class classification_dataset(Dataset):
    def __init__(self,X,Z,bs=1.0,kappa=1,val_rate = 0.01):
        super(classification_dataset, self).__init__()
        self.a_m = [] #m'=m-1, m. m'=2 -> m=3 ->
        self.a_0 = []
        self.n=X.shape[0]
        self.mask = np.array([False] * self.n)
        val_size = round(val_rate*self.n)
        if (self.n-val_size)%2!=0:
            val_size+=1
        self.mask[0:val_size] = True
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
        ref_bs = int(round(self.bs_perc * self.X_joint.shape[0]))
        self.bs =ref_bs


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

        ref_bs = int(round(self.bs_perc * self.X_joint.shape[0]))
        self.bs =ref_bs

class dataset_MI_TRE(Dataset):
    def __init__(self,X,Z,m,p=1,bs=1.0,val_rate = 0.01):
        self.m = m
        self.a_m = [(k/m)**p for k in range(1,m)] #m'=m-1, m. m'=2 -> m=3 ->
        self.a_0 = [(1-el**2)**0.5 for el in self.a_m]
        self.n=X.shape[0]
        mask = np.array([False] * self.n)
        val_size = round(val_rate*self.n)
        if (self.n-val_size)%2!=0:
            val_size+=1
        mask[0:val_size] = True
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
        ref_bs = int(round(self.bs_perc * self.X_joint.shape[0]))
        self.bs =ref_bs

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

class px_categorical(torch.nn.Module):
    def __init__(self,X_cat_train_data):
        super(px_categorical, self).__init__()
        self.denom = X_cat_train_data.shape[0]
        self.d = X_cat_train_data.shape[1]
        self.unique_each_dim = []
        for i in range(self.d):
            _,counts =torch.unique(X_cat_train_data[:,i],return_counts=True)
            prob_vec = counts/self.denom
            self.unique_each_dim.append(counts.shape[0])
            setattr(self,f'prob_vec_{i}',prob_vec)

    def forward(self,X_cat):
        probs = []
        for i in range(self.d):
            idx = X_cat[:,i].long()
            ref = getattr(self,f'prob_vec_{i}')
            probs.append(ref[idx])
        return torch.stack(probs,dim=1)

class p_x_z_net_cat(torch.nn.Module):
    def __init__(self,x_unique,d,cat_marker,cat_size_list,f=12,k=2):
        super(p_x_z_net_cat, self).__init__()
        self.dim = len(x_unique)
        for i,el in enumerate(x_unique):
            setattr(self,f'pxz_{i}',MLP(d,cat_marker,cat_size_list,f,k,o=el))

    def forward(self,Z):
        output_list = []
        for i in range(self.dim):
            net = getattr(self,f'pxz_{i}')
            output = net.pass_through(Z)
            output_list.append(output)
        return output_list

class cat_density_ratio(torch.nn.Module):
    def __init__(self,X_cat_train_data,d,cat_marker,cat_size_list,f=12,k=2):
        super(cat_density_ratio, self).__init__()
        self.cat_X = px_categorical(X_cat_train_data=X_cat_train_data)
        self.x_unique = self.cat_X.unique_each_dim
        self.cat_px_z = p_x_z_net_cat(x_unique=self.x_unique,d=d,cat_marker=cat_marker,cat_size_list=cat_size_list,f=f,k=k)

    def get_pxz_output(self,Z):
        return self.cat_px_z(Z)

    def get_pxz_output_prob(self,X, Z):
        with torch.no_grad():
            o = self.cat_px_z(Z)
            base = 1.0
            for i,el in enumerate(o):
                idx = X[:,i].long()
                el = torch.softmax(el,dim=1)
                res = torch.gather(el, 1, idx.unsqueeze(-1))
                base*=res
        return base

    def get_w(self,X, Z,X_cont):
        p_x = torch.prod(self.cat_X(X),dim=1)
        p_xz = self.get_pxz_output_prob(X,Z)
        w = p_x.squeeze()/p_xz.squeeze()
        return w

class cat_density_ratio_conditional(torch.nn.Module):
    def __init__(self,X_cat_train_data,d_x_cont,d_z,cat_marker,cat_size_list,f=12,k=2):
        super(cat_density_ratio_conditional, self).__init__()
        self.cat_X = px_categorical(X_cat_train_data=X_cat_train_data)
        self.x_unique = self.cat_X.unique_each_dim
        self.cat_px_z = p_x_z_net_cat(x_unique=self.x_unique,d=d_z,cat_marker=cat_marker,cat_size_list=cat_size_list,f=f,k=k)
        self.cat_px_x = p_x_z_net_cat(x_unique=self.x_unique,d=d_x_cont,cat_marker=torch.tensor([False]*d_x_cont),cat_size_list=[],f=f,k=k)

    def get_pxz_output(self,Z):
        return self.cat_px_z(Z)

    def get_pxx_output(self,X_cont):
        return self.cat_px_x(X_cont)

    def get_output_prob(self, model, X_cat, Z):
        with torch.no_grad():
            o = model(Z)
            base = 1.0
            for i,el in enumerate(o):
                idx = X_cat[:,i].long()
                el = torch.softmax(el,dim=1)
                res = torch.gather(el, 1, idx.unsqueeze(-1))
                base*=res
        return base

    def get_w(self,X_cat, Z,X_cont):
        p_xz = self.get_output_prob(self.cat_px_z,X_cat, torch.cat([Z,X_cont],dim=1))
        p_xx = self.get_output_prob(self.cat_px_x,X_cat, X_cont)
        w = p_xx.squeeze()/p_xz.squeeze()
        return w

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
        val_size = round(val_rate*self.n)
        if (self.n-val_size)%2!=0:
            val_size+=1
        mask[0:val_size] = True
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
        ref_bs = int(round(self.bs_perc * self.X_joint.shape[0]))
        # if self.X_joint.shape[0]%ref_bs!=0:
        #     for i in range(8,2*ref_bs):
        #         if self.X_joint.shape[0]%i==0:
        #             self.bs=i
        #             break
        # else:
        self.bs =ref_bs
class chunk_iterator(): #joint = pos, pom = neg
    def __init__(self,X_joint,Z_joint,X_pom,Z_pom,shuffle,batch_size,kappa=10,TRE=False,a_0=[],a_m=[],mode='train'):
        self.mode = mode
        self.X_joint = X_joint
        self.Z_joint = Z_joint
        self.X_pom = X_pom
        self.Z_pom = Z_pom
        self.x_binary = get_binary_mask(self.X_joint)
        self.z_binary = get_binary_mask(self.Z_joint)
        self.shuffle = shuffle
        self.batch_size = batch_size if batch_size!=0 else self.X_joint.shape[0]
        self.n_joint = self.X_joint.shape[0]
        self.n_pom = self.X_pom.shape[0]
        self.chunks_joint = int(round(self.n_joint / self.batch_size)) + 2
        self.perm_joint = torch.randperm(self.n_joint)
        self.kappa=kappa
        if self.shuffle:
            self.X_joint = self.X_joint[self.perm_joint,:]
            self.Z_joint = self.Z_joint[self.perm_joint,:]
        if self.mode=='train':
            self.it_X = torch.chunk(self.X_joint,self.chunks_joint)[:-1]
            self.it_Z = torch.chunk(self.Z_joint,self.chunks_joint)[:-1]
        elif self.mode=='val':
            val_n = min(self.n_joint,self.n_pom)
            self.chunks_joint = int(round(val_n / self.batch_size)) + 1
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

    def generate_pom_samples_old(self, n_):
        vec = np.random.choice(self.n_pom, n_, replace=False)
        c, d = self.X_pom[vec, :], self.Z_pom[:n_]
        return c, d

    def generate_pom_samples(self,n_):
        vec=np.random.choice(self.n_pom,n_,replace=True)
        perm_vec = np.random.permutation(vec)
        c, d = self.X_pom[vec, :], self.Z_pom[perm_vec,:]
        return c,d
    def __next__(self):
        ''''Returns the next value from team object's lists '''
        if self.mode=='train':
            if self._index < self.true_chunks:
                a,b = self.it_X[self._index],self.it_Z[self._index]
                n_ = a.shape[0]*self.kappa
                c,d=self.generate_pom_samples(n_)
                data_out = [torch.cat([c, d], dim=1)]
                if self.TRE:
                    bin_a = a[:, self.x_binary]
                    bin_c = c[:, self.x_binary]
                    disc_dim = bin_a.shape[1]
                    m = len(self.a_m)

                    for i,(a_0, a_m) in enumerate(zip(self.a_0, self.a_m)):
                        transition_x = a_0 * c + a_m * a
                        if disc_dim<2:
                            fac = np.round(i/m)
                            mix_discrete = (1-fac)*bin_c + bin_a*fac
                        else:
                            fac_a = round(i/m * disc_dim)
                            fac_c = disc_dim-fac_a
                            mix_comp_list = []
                            if fac_c>0:
                                mix_comp_list.append(bin_c[:,:fac_c])
                            if fac_a>0:
                                mix_comp_list.append(bin_c[:,:fac_a])
                            mix_discrete = torch.cat(mix_comp_list,dim=1)
                        transition_x[:,self.x_binary] = mix_discrete  #cant have continous transition for binary variables
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

class chunk_iterator_cat(): #joint = pos, pom = neg
    def __init__(self,X_joint,Z_joint,shuffle,batch_size,mode='train'):
        self.mode = mode
        self.X_joint = X_joint
        self.Z_joint = Z_joint
        self.x_binary = get_binary_mask(self.X_joint)
        self.z_binary = get_binary_mask(self.Z_joint)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n_joint = self.X_joint.shape[0]
        self.chunks_joint = self.n_joint // batch_size + 1
        self.perm_joint = torch.randperm(self.n_joint)
        if self.shuffle:
            self.X_joint = self.X_joint[self.perm_joint,:]
            self.Z_joint = self.Z_joint[self.perm_joint,:]

        if self.mode=='train':
            self.it_X = torch.chunk(self.X_joint,self.chunks_joint)
            self.it_Z = torch.chunk(self.Z_joint,self.chunks_joint)
        elif self.mode=='val':
            val_n = self.n_joint
            self.chunks_joint = val_n // batch_size + 1
            self.X_joint = self.X_joint[:val_n,:]
            self.Z_joint = self.Z_joint[:val_n,:]
            self.it_X = torch.chunk(self.X_joint, self.n_joint)
            self.it_Z = torch.chunk(self.Z_joint, self.n_joint)

        self.true_chunks = len(self.it_X)
        self._index = 0

    def __next__(self):
        ''''Returns the next value from team object's lists '''
        if self.mode=='train':
            if self._index < self.true_chunks:
                a,b = self.it_X[self._index],self.it_Z[self._index]
                self._index += 1
                return a,b
            raise StopIteration

        else:
            if self._index < self.true_chunks:
                a,b = self.it_X[self._index],self.it_Z[self._index]
                self._index += 1
                return a,b
            raise StopIteration

    def __len__(self):
        return self.true_chunks

class cat_dataloader():
    def __init__(self,dataset,bs_ratio,shuffle=False):
        self.dataset = dataset
        self.dataset.set_mode('train')
        self.bs_ratio = bs_ratio
        self.batch_size =dataset.bs
        self.shuffle = shuffle
        self.n = self.dataset.X.shape[0]
        self.len=self.n//self.batch_size+1

    def __iter__(self):
        if self.dataset.mode=='train':
            self.batch_size = self.dataset.bs
        else:
            self.batch_size = self.dataset.X.shape[0]//5
        return chunk_iterator_cat(X_joint=self.dataset.X,Z_joint=self.dataset.Z,shuffle=self.shuffle,batch_size=self.batch_size,
                                  mode=self.dataset.mode,
                                  )
    def __len__(self):
        if self.dataset.mode=='train':
            self.batch_size = self.dataset.bs
        else:
            self.batch_size = self.dataset.X.shape[0]//5
        return len(chunk_iterator_cat(X_joint=self.dataset.X,Z_joint=self.dataset.Z,shuffle=self.shuffle,batch_size=self.batch_size,
                                  mode=self.dataset.mode,
                                  ))

class NCE_dataloader():
    def __init__(self,dataset,bs_ratio,shuffle=False,kappa=10,TRE=False):
        self.dataset = dataset
        self.dataset.set_mode('train')
        self.batch_size = self.dataset.bs
        self.shuffle = shuffle
        self.n = self.dataset.X_joint.shape[0]
        self.len=self.n//self.batch_size+1
        self.kappa=kappa
        self.TRE=TRE
    def __iter__(self):
        if self.dataset.mode=='train':
            self.batch_size = self.dataset.bs
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
            self.batch_size = self.dataset.bs
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
