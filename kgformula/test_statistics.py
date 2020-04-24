import torch
import gpytorch
from torch.distributions import Normal,StudentT,Bernoulli,Beta,Uniform,Exponential
import numpy as np
from pykeops.torch import LazyTensor,Genred
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn import metrics
import torch.nn as nn
try:
    from torch.cuda.amp import GradScaler,autocast
except Exception as e:
    print(e)
    print('Install nightly pytorch for mixed precision')

def NCE_objective(true_preds,fake_preds):
    _err = -torch.log(true_preds) - (1.-fake_preds).log().mean(dim=1)
    return _err.mean()

def NCE_objective_stable(true_preds,fake_preds):
    _err = -(true_preds-torch.log(true_preds.exp()+1)-torch.log(fake_preds.exp()+1).mean(dim=1))
    return _err.mean()

def auc_check(y_pred,Y):
    with torch.no_grad():
        y_pred = (y_pred.float() > 0.5).cpu().float().numpy()
        fpr, tpr, thresholds = metrics.roc_curve(Y.cpu().numpy(), y_pred, pos_label=1)
        auc =  metrics.auc(fpr, tpr)
        return auc

class keops_RBFkernel(torch.nn.Module):
    def __init__(self,ls,x,y=None,device_id=0):
        super(keops_RBFkernel, self).__init__()
        self.device_id = device_id
        self.raw_lengthscale = torch.nn.Parameter(ls,requires_grad=False).contiguous()
        self.raw_lengthscale.requires_grad = False
        self.register_buffer('x', x.contiguous())
        self.shape = (x.shape[0],x.shape[0])
        if y is not None:
            self.register_buffer('y',y.contiguous())
        else:
            self.y = x
        self.gen_formula = None

    def get_formula(self,D,ls_size):
        aliases = ['G_0 = Pm(0, ' + str(ls_size) + ')',
                   'X_0 = Vi(1, ' + str(D) + ')',
                   'Y_0 = Vj(2, ' + str(D) + ')',
                   ]
        formula = 'Exp(-G_0*SqNorm2(X_0 - Y_0))'
        return formula,aliases

    def forward(self):
        if self.gen_formula is None:
            self.formula, self.aliases = self.get_formula(D=self.x.shape[1], ls_size=self.raw_lengthscale.shape[0])
            self.gen_formula = Genred(self.formula, self.aliases, reduction_op='Sum', axis=1, dtype='float32')
        return self.gen_formula(*[self.raw_lengthscale,self.x,self.y],backend='GPU',device_id=self.device_id)

def get_i_not_j_indices(n):
    vec = np.arange(0, n)
    vec_2 = np.arange(0, int(n ** 2), n+1)
    list_np = np.array(np.meshgrid(vec, vec)).T.reshape(-1, 2)
    list_np = np.delete(list_np, vec_2, axis=0)
    return list_np

class MLP_feature_map(torch.nn.Module):
    def __init__(self,d,f=12,k=2,o=1):
        super(MLP_feature_map, self).__init__()
        self.model = nn.ModuleList()
        self.model.append(nn.Linear(d, f))
        self.model.append(nn.Tanh())
        for i in range(k):
            self.model.append(nn.Linear(f, f))
            self.model.append(nn.Tanh())
        self.model.append(nn.Linear(f, o))
        # self.model.append(nn.Tanh())

    def forward(self,x):
        for l in self.model:
            x = l(x)
        return x

class MLP(torch.nn.Module):
    def __init__(self,d,f=12,k=2,o=1):
        super(MLP, self).__init__()
        self.model = nn.ModuleList()
        self.model.append(nn.Linear(d, f))
        self.model.append(nn.Tanh())
        for i in range(k):
            self.model.append(nn.Linear(f, f))
            self.model.append(nn.Tanh())
        self.model.append(nn.Linear(f, o))
        self.model.append(nn.Tanh())

    def forward(self,X,Z):
        x = torch.cat([X,Z],dim=1)
        for l in self.model:
            x = l(x)
        return x

    def forward_predict(self,X,Z):
        return self.forward(X,Z)

    def get_w(self, x, y):
        pred = torch.sigmoid(self.forward(x,y))
        return (1-pred)/pred


class HSIC_MLP_classifier(torch.nn.Module):
    def __init__(self,x_data_params,y_data_params):
        super(HSIC_MLP_classifier, self).__init__()
        self.MLP_x_feature_map = MLP_feature_map(**x_data_params)
        self.MLP_y_feature_map = MLP_feature_map(**y_data_params)

    def MSE_obj(self,xy,xy_ref):
        return (xy-xy_ref).square().sum(1)

    def calculate_HSIC(self,x,y,x_ref,y_ref):
        xy = self.MLP_x_feature_map(x)*self.MLP_y_feature_map(y)
        xy_ref = self.MLP_x_feature_map(x_ref)*self.MLP_y_feature_map(y_ref)
        v = self.MSE_obj(xy,xy_ref)
        return v

    def forward(self,x,y,x_ref,y_ref):
        return (1.-torch.exp(-self.calculate_HSIC(x,y,x_ref,y_ref)))*0.99

    def forward_predict(self,x,y):
        idx = torch.randperm(y.shape[0])
        return (1.-torch.exp(-self.calculate_HSIC(x,y,x,y[idx])))*0.99

    def get_w(self,x,y):
        idx = torch.randperm(y.shape[0])
        return 1./(torch.exp(self.calculate_HSIC(x,y,x,y[idx]))-1)

class HSIC_MLP_classifier_variant(HSIC_MLP_classifier):
    def __init__(self,x_data_params,y_data_params):
        super(HSIC_MLP_classifier_variant, self).__init__(x_data_params,y_data_params)

    def forward(self, x, y, x_ref, y_ref):
        return torch.log(self.calculate_HSIC(x,y,x_ref,y_ref))

    def forward_predict(self, x, y):
        idx = torch.randperm(y.shape[0])
        return torch.log(self.calculate_HSIC(x, y, x, y[idx]))

    def get_w(self, x, y):
        idx = torch.randperm(y.shape[0])
        return 1. /self.calculate_HSIC(x, y, x, y[idx])
class logistic_regression(torch.nn.Module):
    def __init__(self,d):
        super(logistic_regression, self).__init__()
        self.W = torch.nn.Linear(in_features=d,  out_features=1,bias=True)

    def forward(self,x):
        return self.W(x)

    def logistic_forward(self,x):
        return torch.nn.functional.sigmoid(self.W(x))

class classification_dataset(Dataset):
    def __init__(self,X,Z,bs=1.0,kappa=1):
        super(classification_dataset, self).__init__()
        self.X = X
        self.Z = Z
        self.bs = int(round(bs*self.X.shape[0]))
        self.device = X.device
        self.kappa = kappa
        self.sample_indices_base = np.arange(self.X.shape[0])
        self.HSIC_mode = False

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
        if self.HSIC_mode:
            HSIC_ref_indices_true = self.sample_no_replace(fake_set,1,False)
            HSIC_ref_indices_fake = self.sample_no_replace(fake_set,self.kappa,False)
        else:
            HSIC_ref_indices_true = None
            HSIC_ref_indices_fake = None
        return true_indices,fake_indices,HSIC_ref_indices_true,HSIC_ref_indices_fake

    def get_sample(self):
        T,F,HSIC_T,HSIC_F = self.get_indices()
        if self.HSIC_mode:
            return self.X[T],self.Z[T],self.X[T.repeat(self.kappa)],self.Z[F],self.Z[HSIC_T],self.Z[HSIC_F]
        else:
            return self.X[T],self.Z[T],self.X[T.repeat(self.kappa)],self.Z[F],None,None

class density_estimator():
    def __init__(self, x, z, est_params=None, cuda=False, device=0, type='linear'):
        self.failed = False
        self.x = x
        self.z = z
        self.cuda = cuda
        self.n = self.x.shape[0]
        self.device = device
        self.est_params = est_params
        self.type = type
        self.kernel_base = gpytorch.kernels.Kernel()

        if type=='linear':
            self.alpha = est_params['alpha']
            self.diag = est_params['reg_lambda']*torch.eye(self.n)
            if self.cuda:
                self.diag = self.diag.cuda(self.device)
            self.linear_x_of_z()
            self.get_w_kdre()

        elif type=='gp':
            self.alpha = est_params['alpha']
            self.diag = est_params['reg_lambda']*torch.eye(self.n)
            if self.cuda:
                self.diag = self.diag.cuda(self.device)
            self.kernel_ls_init('kernel_tmp', self.z)
            self.gp_x_of_z()
            self.get_w_kdre()

        elif type=='semi':
            self.semi_cheat_x_of_z()

        elif type == 'kmm':
            self.diag = est_params['reg_lambda']*torch.eye(self.n)
            if self.cuda:
                self.diag = self.diag.cuda(self.device)
            self.kernel_mean_matching()

        elif type == 'classifier':
            dataset = self.create_classification_data()
            self.model = MLP(d=dataset.X.shape[1]+dataset.Z.shape[1],f=self.est_params['width'],k=self.est_params['layers']).to(self.x.device)
            self.w = self.train_classifier(dataset)

        elif type == 'HSIC_classifier':
            dataset = self.create_classification_data()
            dataset.HSIC_mode = True
            self.model = HSIC_MLP_classifier_variant(x_data_params=self.est_params['x_params'],y_data_params=self.est_params['y_params']).to(self.x.device)
            self.w = self.train_classifier(dataset)

    def retrain(self,x,z):
        self.x = x
        self.z = z
        if self.type == 'HSIC_classifier':
            dataset = self.create_classification_data()
            self.w = self.train_classifier(dataset)
        elif self.type == 'classifier':
            dataset = self.create_classification_data()
            self.w = self.train_classifier(dataset)

    def forward_pred(self,X,Z,):
        if self.type == 'HSIC_classifier':
            pred_T = self.model.forward_predict(X,Z)
        elif self.type == 'classifier':
            pred_T = self.model.forward_predict(X,Z)
        return pred_T.squeeze()

    def forward_func(self,X,Z,X_fake,Z_fake,Z_hsic_T,Z_hsic_F,loss_func):
        if self.type == 'HSIC_classifier':
            pred_T = self.model(X,Z,X,Z_hsic_T)
            pred_F = self.model(X_fake,Z_fake,X_fake,Z_hsic_F)
        elif self.type == 'classifier':
            pred_T = self.model(X,Z)
            pred_F = self.model(X_fake,Z_fake)
        pred_F = pred_F.view(pred_T.shape[0],-1)
        return loss_func(pred_T,pred_F)

    def train_classifier(self,dataset):
        # dataloader = DataLoader(dataset,batch_size=self.est_params['batch_size'],shuffle=True)
        loss_func = NCE_objective_stable
        opt = torch.optim.Adam(self.model.parameters(),lr=self.est_params['lr'])
        if self.est_params['mixed']:
            scaler = GradScaler()
        # y_test = torch.cat([torch.ones(dataset.X.shape[0]),torch.zeros(dataset.X.shape[0])])
        counter = 0
        best = np.inf
        idx = torch.randperm(dataset.X.shape[0])
        for i in range(self.est_params['max_its']):
            X_true,Z_true,X_fake,Z_fake,Z_HSIC_T,Z_HSIC_F = dataset.get_sample()
            opt.zero_grad()
            if self.est_params['mixed']:
                with autocast():
                    l = self.forward_func(X_true,Z_true,X_fake,Z_fake,Z_HSIC_T,Z_HSIC_F,loss_func)
                scaler.scale(l).backward()
                scaler.step(opt)
                scaler.update()
            else:
                l = self.forward_func(X_true, Z_true, X_fake, Z_fake, Z_HSIC_T, Z_HSIC_F,loss_func)
                l.backward()
                opt.step()

            if i%(self.est_params['max_its']//50)==0:
                # print(l)
                with torch.no_grad():
                    pred_T = self.forward_pred(dataset.X,dataset.Z)
                    pred_F = self.forward_pred(dataset.X,dataset.Z[idx])
                    pred_F = pred_F.view(pred_T.shape[0], -1)
                    # auc = self.forward_func(dataset.X, dataset.Z, dataset.X, dataset.Z[idx], Z_HSIC_T, Z_HSIC_F, loss_func)
                    # print(pred_T)
                    # print(pred_F)
                    # auc = auc_check(torch.cat([pred_T,pred_F]),y_test)
                    auc = loss_func(pred_T,pred_F)
                    print(f'logloss epoch {i}: {auc}')
                    if auc.item()<best:
                        best = auc.item()
                        counter=0
                    else:
                        counter+=1

                    # print(f'auc epoch {j}: {auc}')
            if counter>self.est_params['kill_counter']:
                print('stopped improving, stopping')
                break
        with torch.no_grad():
            w = self.model.get_w(dataset.X,dataset.Z)
        return w
    def create_classification_data(self):
        with torch.no_grad():
            self.kappa = self.est_params['kappa']
        return classification_dataset(self.x,self.z,bs=self.est_params['bs_ratio'],kappa=self.kappa)

    def kernel_mean_matching(self):
        with torch.no_grad():
            list_idx = torch.from_numpy(get_i_not_j_indices(self.n)) #Seems to be working alright!
            torch_idx_x,torch_idx_z = list_idx.unbind(dim=1)
            data_extended = torch.cat([self.x[torch_idx_x],self.z[torch_idx_z]],dim=1)
            data = torch.cat([self.x,self.z],dim=1)
            self.kernel_ls_init('kernel_up', data)
            ls = self.get_median_ls_XY(data,data_extended)
            y = self.kernel_ls_init_keops(ls=ls,data=data,data_2=data_extended)
            self.w,_ = torch.solve(y,(self.kernel_up+self.diag))

    def get_w_kdre(self):
        self.kernel_ls_init('kernel_up', self.x)
        self.kernel_ls_init('kernel_down', self.x, self.down_estimator)
        with torch.no_grad():
            self.kernel_down = self.kernel_down.evaluate()
            self.h_hat = self.kernel_up.mean(dim=1,keepdim=True)
            self.H = self.alpha/self.n * self.kernel_up@self.kernel_up + (1-self.alpha)/self.n * (self.kernel_down@self.kernel_down) + self.diag
            self.theta,_ = torch.solve(self.h_hat, self.H)
            self.w = self.kernel_up@self.theta

    def return_weights(self):
        return self.w.squeeze()

    def linear_x_of_z(self):
        down = torch.cat([self.z, torch.ones_like(self.z)], dim=1)
        with torch.no_grad():
            self.down_estimator = down@(torch.inverse(down.t()@down) @ (down.t() @ self.x))

    def semi_cheat_x_of_z(self):
        with torch.no_grad():
            self.linear_x_of_z()
            res = self.z - self.down_estimator
            p_1 = Normal(0,scale=res.var()**0.5)
            p_2 = Normal(0, scale=self.z.var() ** 0.5)
            self.w = (p_2.log_prob(self.z - self.z.mean()) - p_1.log_prob(res)).exp()

    def gp_x_of_z(self):
        with torch.no_grad():
            s,_ = torch.solve(self.x, self.kernel_tmp + self.diag)
            self.down_estimator = self.kernel_tmp@s

    def get_median_ls_XY(self,X,Y):
        with torch.no_grad():
            if X.shape[0]>5000:
                X = X[torch.randperm(5000),:]
            if Y.shape[0]>5000:
                Y = Y[torch.randperm(5000),:]
            d = self.kernel_base.covar_dist(x1=X,x2=Y)
            ret = torch.sqrt(torch.median(d[d > 0]))
            return ret

    def get_median_ls(self,X):
        with torch.no_grad():
            d = self.kernel_base.covar_dist(x1=X,x2=X)
            ret = torch.sqrt(torch.median(d[d > 0]))
            return ret

    def kernel_ls_init_keops(self,ls,data,data_2=None):
        with torch.no_grad():
            ker = keops_RBFkernel(ls=1/ls.unsqueeze(0),x=data,y=data_2,device_id=self.device)
            return ker()/(self.n-1)

    def kernel_ls_init(self,name,data,data_2=None,ls=None):
        ker = gpytorch.kernels.RBFKernel()
        if ls is None:
            ls = self.get_median_ls(data)
        ker._set_lengthscale(ls)
        if self.cuda:
            ker = ker.cuda(self.device)
            if data_2 is None:
                setattr(self,name,ker(data).evaluate())
            else:
                setattr(self,name,ker(data,data_2))
        return ls

class weighted_stat(): #HAPPY MISTAKE?!?!??!?!?!?!?!?
    def __init__(self,X,Y,Z,w,do_null = True,reg_lambda=1e-3,cuda=False,device=0,half_mode=False):
        with torch.no_grad():
            self.device = device
            self.cuda = cuda
            self.n = X.shape[0]
            self.H = torch.ones(*(self.n, 1)) * (1 - 1 / self.n)
            self.H_2 = torch.eye(self.n)-2*torch.ones(*(self.n, self.n))/self.n
            self.H_4 = torch.eye(self.n)-4*torch.ones(*(self.n, self.n))/self.n
            self.one_n_1 = torch.ones(*(self.n,1))
            self.ones= torch.ones(*(self.n,self.n))
            self.half_mode = half_mode
            if cuda:
                self.H = self.H.cuda(device)
                self.H_2 = self.H_2.cuda(device)
                self.H_4 = self.H_4.cuda(device)
                self.ones = self.ones.cuda(device)
                self.one_n_1 = self.one_n_1.cuda(device)
            else:
                self.device = 'cpu'
            self.X = X
            self.Y = Y
            self.Z = Z
            self.w = w.unsqueeze(-1)
            self.W = self.w@self.w.t()
            self.W = self.W/self.n
            self.do_null=do_null
            self.kernel_base = gpytorch.kernels.Kernel()
            self.reg_lambda = reg_lambda
            for name,data in zip(['X','Y'],[self.X,self.Y]):
                self.kernel_ls_init(name,data)
            if self.half_mode:
                self.H = self.H.half()
                self.w = self.w.half()
                self.W = self.W.half()
            self.X_ker = self.kernel_X
            self.Y_ker = self.kernel_Y
            self.center_X = (self.X_ker@self.H)*self.w
            self.center_Y = self.kernel_Y@self.H

    def kernel_ls_init(self,name,data):
        setattr(self, f'ker_obj_{name}', gpytorch.kernels.RBFKernel().cuda(self.device) if self.cuda else gpytorch.kernels.RBFKernel())
        ls = self.get_median_ls(data)
        getattr(self,f'ker_obj_{name}')._set_lengthscale(ls)
        if self.half_mode:
            setattr(self,f'kernel_{name}',getattr(self,f'ker_obj_{name}')(data).evaluate().half())
        else:
            setattr(self,f'kernel_{name}',getattr(self,f'ker_obj_{name}')(data).evaluate())

    def get_median_ls(self,X):
        with torch.no_grad():
            d = self.kernel_base.covar_dist(x1=X,x2=X)
            ret = torch.sqrt(torch.median(d[d > 0]))
            return ret

class weighted_statistic_new(weighted_stat):
    def __init__(self,X,Y,Z,w,do_null = True,reg_lambda=1e-3,cuda=False,device=0):
        super(weighted_statistic_new, self).__init__(X, Y, Z, w, do_null, reg_lambda, cuda, device)
        with torch.no_grad():
            self.sum_mean_X = self.X_ker.mean()
            self.X_ker_H_4 = self.X_ker@self.H_4
            self.X_ker_H_2= self.X_ker@self.H_2
            self.X_ker_n_1 = self.X_ker@self.one_n_1
            self.X_ker_n_1 = self.X_ker_n_1/self.n
            self.X_ker_ones =self.X_ker@self.ones
            self.X_ker_ones =self.X_ker_ones/self.n

            self.sum_mean_Y = self.Y_ker.mean()
            self.Y_ker_H_4 = self.Y_ker@self.H_4
            self.Y_ker_H_2= self.Y_ker@self.H_2
            self.Y_ker_n_1 = self.Y_ker@self.one_n_1
            self.Y_ker_ones =self.Y_ker@self.ones
            self.Y_ker_n_1 = self.Y_ker_n_1/self.n
            self.Y_ker_ones =self.Y_ker_ones/self.n

            self.term_1 = 0.5*self.W*self.X_ker_H_4
            self.term_2 = 0.5*self.W*self.X
            self.term_3 = 2*self.W*self.X_ker_ones
            self.term_4 = 2*self.W
            self.term_5 = self.W*self.sum_mean_X
            self.term_6 = self.W*self.sum_mean_Y*self.X_ker_H_2
            self.term_7 = self.W*self.sum_mean_Y*self.sum_mean_X

    def permutation_calculate_weighted_statistic(self):
        with torch.no_grad():
            idx = torch.randperm(self.n)
            y = self.Y[idx]
            Y_ker = self.ker_obj_Y(y).evaluate()
            Y_ker_H_4 = Y_ker @ self.H_4
            Y_ker_H_2 = Y_ker @ self.H_2
            Y_ker_n_1 = Y_ker @ self.one_n_1
            Y_ker_n_1 = Y_ker_n_1/self.n
            Y_ker_ones = Y_ker @ self.ones
            Y_ker_ones = Y_ker_ones/self.n
            test_stat = self.term_1 * Y_ker + self.term_2 * Y_ker_H_4 + self.term_3 * Y_ker_ones + self.term_4 * (
                        self.X_ker_n_1 @ Y_ker_n_1.t()) + self.term_5 * Y_ker_H_2 + self.term_6 + self.term_7
            return test_stat.sum()

    def calculate_weighted_statistic(self):
        with torch.no_grad():
            test_stat = self.term_1 * self.Y_ker + self.term_2 * self.Y_ker_H_4 + self.term_3 * self.Y_ker_ones + self.term_4 * (
                    self.X_ker_n_1 @ self.Y_ker_n_1.t()) + self.term_5 * self.Y_ker_H_2 + self.term_6 + self.term_7
            return test_stat.sum()






