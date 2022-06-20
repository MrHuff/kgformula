import torch.nn
import random
from kgformula.networks import *
from kgformula.kernels import *
import os
import copy
from sklearn.preprocessing import KBinsDiscretizer



def get_binary_mask(X):
    #TODO: rewrite this a bit
    dim = X.shape[1]
    mask_ls = [0] * dim
    label_size = []
    for i in range(dim):
        x = X[:, i]
        un_el = x.unique()
        mask_ls[i] = un_el.numel() <= 5
        if mask_ls[i]:
            label_size.append(un_el.tolist())

    return torch.tensor(mask_ls),label_size
class HSIC_independence_test():
    def __init__(self,X,Y,n_samples):
        self.X = X
        self.Y = Y
        self.n = self.X.shape[0]
        if self.n>5000:
            idx = torch.randperm(2500)
            self.X = self.X[idx,:]
            self.Y = self.Y[idx,:]
            self.n = 2500
        self.kernel_base = Kernel()
        self.kernel_ls_init('ker_X',self.X)
        self.kernel_ls_init('ker_Y',self.Y)
        self.ker_X_eval = self.ker_X(self.X).evaluate()
        self.ker_Y_eval = self.ker_Y(self.Y).evaluate()
        self.ker_X_eval_ones = (self.ker_X_eval@torch.ones(*(self.n,1),device=self.X.device)).repeat(1,self.n)
        self.ker_Y_eval_ones = (self.ker_Y_eval@torch.ones(*(self.n,1),device=self.X.device)).repeat(1,self.n)
        self.X_cache = self.ker_X_eval-self.ker_X_eval_ones
        with torch.no_grad():
            self.ref_metric = torch.mean(self.X_cache*(self.ker_Y_eval-self.ker_Y_eval_ones)).item()
            self.p_val = self.calc_permute(self.X,self.Y,n_samples=n_samples)

    def get_median_ls(self,X):
        with torch.no_grad():
            if self.n>5000:
                idx = torch.randperm(2500)
                X = X[idx,:]
            d = self.kernel_base.covar_dist(x1=X,x2=X)
            ret = torch.sqrt(torch.median(d[d >= 0]))
            return ret

    @staticmethod
    def calculate_pval(bootstrapped_list, test_statistic):
        pval_right = 1 - 1 / (bootstrapped_list.shape[0] + 1) * (1 + (bootstrapped_list <= test_statistic).sum())
        pval_left = 1 - pval_right
        pval = 2 * min([pval_left.item(), pval_right.item()])
        return pval

    def kernel_ls_init(self,name,data,ls=None):
        ker = RBFKernel()
        if ls is None:
            ls = self.get_median_ls(data)
        ker._set_lengthscale(ls)
        ker = ker.to(data.device)
        setattr(self,name,ker)

    def calculate_HSIC(self,idx):
        Y_ker = self.ker_Y_eval[idx,:]
        Y_ker = Y_ker[:,idx]
        return torch.mean(self.X_cache*(Y_ker-self.ker_Y_eval_ones[idx,:]))

    def calc_permute(self,X,Y,n_samples):
        self.bootstrapped=[]
        with torch.no_grad():
            for i in range(n_samples):
                idx = torch.randperm(self.X.shape[0])
                self.bootstrapped.append(self.calculate_HSIC(idx).item())
        self.bootstrapped = np.array(self.bootstrapped)
        return self.calculate_pval(self.bootstrapped,self.ref_metric)

def hsic_test(X,Y,n_sample = 250):
    test = HSIC_independence_test(X,Y,n_sample)
    return test.p_val

def hsic_sanity_check_w(w,x,z,n_perms=250):
    _w = w.cpu().squeeze().numpy()
    idx_HSIC = np.random.choice(np.arange(x.shape[0]), x.shape[0], p=_w / _w.sum())
    p_val = hsic_test(x[idx_HSIC, :], z[idx_HSIC, :], n_perms)
    return p_val


def get_i_not_j_indices(n):
    vec = np.arange(0, n)
    vec_2 = np.arange(0, int(n ** 2), n+1)
    list_np = np.array(np.meshgrid(vec, vec)).T.reshape(-1, 2)
    list_np = np.delete(list_np, vec_2, axis=0)
    return list_np

class density_estimator():
    def __init__(self, x, z,x_q,x_full=None,z_full=None, est_params=None, cuda=False, device=0, type='linear',secret_indx=0,cat_cols_z={}):
        #TODO: Configure the cat_cols to make sense

        self.x = x
        self.z = z
        self.cat_x_marker,tmp_2= get_binary_mask(x_full)
        self.x_cont = self.x[:,~self.cat_x_marker]
        self.x_cat = self.x[:,self.cat_x_marker]
        self.has_cat = self.x_cat.shape[1]>0
        self.has_cont = self.x_cont.shape[1]>0
        self.x_q = x_q
        self.x_q_cont = self.x_q[:,~self.cat_x_marker]
        self.x_q_cat = self.x_q[:,self.cat_x_marker]

        """
        cat and cont separation procedure...
        """
        if len(cat_cols_z)==0:
            tmp_3,tmp_4= get_binary_mask(z_full)
        else:
            tmp_3,tmp_4 = torch.tensor(cat_cols_z['indicator']),cat_cols_z['index_lists']

        self.cat_marker = torch.tensor([False] * x_full[:, ~self.cat_x_marker].shape[1] + tmp_3.tolist())
        self.cat_list = tmp_4
        self.cont_marker = ~self.cat_marker

        if self.x_q.dim()==1:
            self.x_q = self.x_q.unsqueeze(-1)
        self.cuda = cuda
        self.n = self.x.shape[0]
        self.device = device
        self.est_params = est_params
        self.type = type
        self.kernel_base = Kernel()
        if self.est_params['separate']:

            if self.type == 'random_uniform':
                self.has_cat = False
                self.w = torch.rand(*(self.x.shape[0],1)).squeeze().cuda(self.device)
            elif self.type == 'ones':
                self.has_cat = False
                self.w = torch.ones(*(self.x.shape[0],1)).squeeze().cuda(self.device)
            else:
                if self.has_cat:
                    if self.has_cont:
                        cond_data = torch.cat([self.z,self.x_cont],dim=1)
                        tmp_zx_1, tmp_zx_2 = get_binary_mask(cond_data)
                        self.d_z = self.z.shape[1]
                        self.kappa=1
                        self.dataset = cat_dataset(X=self.x_cat,Z=cond_data, bs = self.est_params['bs_ratio'],
                                                val_rate = self.est_params['val_rate'])
                        self.dataloader = cat_dataloader(dataset=self.dataset,bs_ratio=self.est_params['bs_ratio'],shuffle=True)
                        self.model_cat = cat_density_ratio_conditional(
                            X_cat_train_data=self.x_cat,
                            d_x_cont=self.x_cont.shape[1],
                            d_z=sum(~tmp_zx_1), cat_size_list=tmp_zx_2, cat_marker=tmp_zx_1,
                                         f=self.est_params['width'], k=self.est_params['layers']).to(self.device)
                        self.train_classifier_categorical()
                    else:
                        self.kappa = 1
                        self.dataset = cat_dataset(X=self.x_cat, Z=self.z, bs=self.est_params['bs_ratio'],
                                                   val_rate=self.est_params['val_rate'])
                        self.dataloader = cat_dataloader(dataset=self.dataset, bs_ratio=self.est_params['bs_ratio'],
                                                         shuffle=True)
                        self.model_cat = cat_density_ratio(
                            X_cat_train_data=self.x_cat,
                            d=sum(~tmp_3), cat_size_list=tmp_4, cat_marker=tmp_3,
                            f=self.est_params['width'], k=self.est_params['layers']).to(self.x_cat.device)
                        self.train_classifier_categorical()

                if self.has_cont:
                    # For all density ratios, mixed z is ok but separated x
                    self.dataset = self.create_classification_data()
                    if self.type in ['NCE','NCE_Q','real_TRE','real_TRE_Q']:
                        self.dataloader = NCE_dataloader(dataset=self.dataset,bs_ratio=self.est_params['bs_ratio'],shuffle=True,kappa=self.kappa,
                                                    TRE=self.type in ['real_TRE','real_TRE_Q'])
                    if self.type == 'NCE':
                        self.model = MLP(d=self.cont_marker.sum().item(),cat_size_list=self.cat_list,cat_marker=self.cat_marker,f=self.est_params['width'],k=self.est_params['layers']).to(self.x.device)
                        self.train_classifier()

                    elif self.type=='NCE_Q':
                        self.model = MLP(d=self.cont_marker.sum().item(),cat_size_list=self.cat_list,cat_marker=self.cat_marker, f=self.est_params['width'],k=self.est_params['layers']).to(self.x.device)
                        self.train_classifier()

                    elif self.type=='real_TRE':
                        self.model = TRE_net(dim=self.cont_marker.sum().item(),
                                             o = 1,
                                             f=self.est_params['width'],
                                             k=self.est_params['layers'],
                                             m = self.est_params['m'],cat_marker=self.cat_marker,cat_size_list=self.cat_list
                                             ).to(self.x.device)
                        self.train_classifier()
                    elif self.type=='real_TRE_Q':
                        self.model = TRE_net(dim=self.cont_marker.sum().item(),
                                             o = 1,
                                             f=self.est_params['width'],
                                             k=self.est_params['layers'],
                                             m = self.est_params['m'],cat_marker=self.cat_marker,cat_size_list=self.cat_list
                                             ).to(self.x.device)
                        self.train_classifier()
                    elif self.type == 'rulsif':
                        self.train_rulsif(self.dataset)
        else:
            if self.type == 'random_uniform':
                self.has_cat = False
                self.w = torch.rand(*(self.x.shape[0], 1)).squeeze().cuda(self.device)
            elif self.type == 'ones':
                self.has_cat = False
                self.w = torch.ones(*(self.x.shape[0], 1)).squeeze().cuda(self.device)
            else:

                """
                further separation procedure
                """
                if len(cat_cols_z) == 0:
                    cat_data = torch.cat([self.x, self.z], dim=1)
                    self.cat_marker, self.cat_list = get_binary_mask(cat_data)
                else:
                    tmp_3, tmp_4 = torch.tensor(cat_cols_z['indicator']), cat_cols_z['index_lists']
                    self.cat_marker = torch.cat([self.cat_x_marker,tmp_3],dim=0)
                    self.cat_list = tmp_2+tmp_4


                self.x_cont = self.x
                self.x_q_cont = self.x_q
                self.dataset = self.create_classification_data()
                if self.type in ['NCE', 'NCE_Q', 'real_TRE', 'real_TRE_Q']:
                    self.dataloader = NCE_dataloader(dataset=self.dataset, bs_ratio=self.est_params['bs_ratio'],
                                                     shuffle=True, kappa=self.kappa,
                                                     TRE=self.type in ['real_TRE', 'real_TRE_Q'])
                if self.type == 'NCE':
                    self.model = MLP(d=self.cont_marker.sum().item(), cat_size_list=self.cat_list,
                                     cat_marker=self.cat_marker, f=self.est_params['width'],
                                     k=self.est_params['layers']).to(self.x.device)
                    self.train_classifier()

                elif self.type == 'NCE_Q':
                    self.model = MLP(d=self.cont_marker.sum().item(), cat_size_list=self.cat_list,
                                     cat_marker=self.cat_marker, f=self.est_params['width'],
                                     k=self.est_params['layers']).to(self.x.device)
                    self.train_classifier()

                elif self.type == 'real_TRE':
                    self.model = TRE_net(dim=self.cont_marker.sum().item(),
                                         o=1,
                                         f=self.est_params['width'],
                                         k=self.est_params['layers'],
                                         m=self.est_params['m'], cat_marker=self.cat_marker, cat_size_list=self.cat_list
                                         ).to(self.x.device)
                    self.train_classifier()
                elif self.type == 'real_TRE_Q':
                    self.model = TRE_net(dim=self.cont_marker.sum().item(),
                                         o=1,
                                         f=self.est_params['width'],
                                         k=self.est_params['layers'],
                                         m=self.est_params['m'], cat_marker=self.cat_marker, cat_size_list=self.cat_list
                                         ).to(self.x.device)
                    self.train_classifier()
                elif self.type == 'rulsif':
                    self.has_cat = False
                    self.train_rulsif(self.dataset)

    def create_classification_data(self):
        self.kappa = self.est_params['kappa']
        if self.type == 'NCE':
            return classification_dataset(X=self.x_cont,
                                   Z=self.z,
                                    bs = self.est_params['bs_ratio'],
                                    kappa = self.kappa,
                                    val_rate = self.est_params['val_rate']
            )
        elif self.type=='NCE_Q':
            return classification_dataset_Q(X=self.x_cont,
                                          Z=self.z,
                                          X_q=self.x_q_cont,
                                          bs=self.est_params['bs_ratio'],
                                          kappa=self.kappa,
                                          val_rate=self.est_params['val_rate']
                                          )
        elif self.type=='real_TRE':
            return dataset_MI_TRE(X=self.x_cont,
                                            Z=self.z,
                                            m = self.est_params['m'],
                                            bs=self.est_params['bs_ratio'],
                                            val_rate=self.est_params['val_rate'],)
        elif self.type=='real_TRE_Q':
            return dataset_MI_TRE_Q(X=self.x_cont,X_q=self.x_q_cont,
                                            Z=self.z,
                                            m = self.est_params['m'],
                                            bs=self.est_params['bs_ratio'],
                                            val_rate=self.est_params['val_rate'],)

        elif self.type=='rulsif':
            return dataset_rulsif(X=self.x_cont,X_q=self.x_q_cont,Z=self.z,)

    def calc_loss(self,loss_func,pf,pt,target):
        if loss_func.__class__.__name__=='standard_bce':
            return loss_func(torch.cat([pt.squeeze(),pf.flatten()]),target)
        elif loss_func.__class__.__name__=='NCE_objective_stable':
            pf = pf.view(pt.shape[0], -1)
            return loss_func(pf,pt)

    def train_rulsif(self,dataset):
        pom,joint,pom_q = dataset.get_data()
        self.model = rulsif(pom=pom,joint=joint).to(self.device)
        self.model.calc_theta()

    def validation_loop(self,epoch):
        self.dataloader.dataset.set_mode('val')
        with torch.no_grad():
            pt = []
            pf = []
            for i,datalist in enumerate(self.dataloader):
                n = self.model.forward_val(datalist[0])
                p = self.model.forward_val(datalist[1])
                pf.append(n)
                pt.append(p)
            pt = torch.cat(pt,dim=0)
            pf = torch.cat(pf,dim=0)
            one_y = torch.ones_like(pt)
            zero_y = torch.zeros_like(pf)
            target = torch.cat([zero_y,one_y])
            logloss = self.calc_loss(self.loss_func,pf,pt,target)
            preds = torch.sigmoid(torch.cat([pf.squeeze(),pt.squeeze()]))
            auc = auc_check(  preds,target.squeeze())
            print(f'logloss epoch {epoch}: {logloss}')
            print(f'auc epoch {epoch}: {auc}')
            self.scheduler.step(logloss)
            if logloss.item()<self.best:
                self.best = logloss.item()
                self.counter=0
                self.best_model = copy.deepcopy(self.model)
            else:
                self.counter+=1
            if self.counter>self.est_params['kill_counter']:
                return True
            else:
                return False

    def train_loop(self):
        self.dataloader.dataset.set_mode('train')
        total_err = 0.
        for i,datalist in enumerate(self.dataloader):
            self.opt.zero_grad()
            l = self.forward_func(list_data=datalist, loss_func=self.loss_func)
            l.backward()
            self.opt.step()
            total_err+=l.item()
        print(f'train err: {total_err/i}')

    def forward_func(self, list_data, loss_func):
        list_preds = self.model(list_data)
        loss = 0
        for preds in list_preds:
            loss += self.calc_loss(loss_func, preds[0], preds[1], [])
        return loss / len(list_preds)

    def categorical_classification_loss(self,output,x):
        loss = 0
        for i,o in enumerate(output):
            sub_x = x[:,i].long()
            l = self.loss_func(o,sub_x)
            loss+=l
        return loss

    def train_classifier_categorical(self):
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.model_cat.parameters(), lr=self.est_params['lr'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt,factor=0.5, patience=1)
        self.counter = 0
        self.best = np.inf
        self.best_model_cat = copy.deepcopy(self.model_cat)
        for epoch in range(self.est_params['max_its']):
            print(f'epoch {epoch+1}')
            self.dataloader.dataset.set_mode('train')
            total_err = 0.
            for i, (x,z) in enumerate(self.dataloader):
                self.opt.zero_grad()
                output = self.model_cat.get_pxz_output(z)
                l_cat = self.categorical_classification_loss(output,x)
                if self.has_cont:
                    output_cont = self.model_cat.get_pxx_output(z[:,self.d_z:])
                    l_cont = self.categorical_classification_loss(output_cont, x)
                else:
                    l_cont  = 0
                l_tot = l_cat + l_cont
                l_tot.backward()
                self.opt.step()
                total_err += l_tot.item()

            self.dataloader.dataset.set_mode('val')
            total_err_val = 0.

            with torch.no_grad():
                for i, (x,z) in enumerate(self.dataloader):
                    output = self.model_cat.get_pxz_output(z)
                    l_cat = self.categorical_classification_loss(output, x)
                    if self.has_cont:
                        output_cont = self.model_cat.get_pxx_output(z[:, self.d_z:])
                        l_cont = self.categorical_classification_loss(output_cont, x)
                    else:
                        l_cont = 0
                    l_tot = l_cat + l_cont
                    total_err_val += l_tot.item()

                self.scheduler.step(total_err_val)
                if total_err_val < self.best:
                    self.best =total_err_val
                    self.counter = 0
                    self.best_model_cat=copy.deepcopy(self.model_cat)
                else:
                    self.counter += 1
                if self.counter > self.est_params['kill_counter']:
                    return True

            print(f'train err: {total_err}')
            print(f'val err: {total_err_val}')

    def train_classifier(self):
        self.loss_func = NCE_objective_stable(self.kappa)
        self.opt = torch.optim.Adam(self.model.parameters(),lr=self.est_params['lr'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt,factor=0.5, patience=1)
        self.counter = 0
        self.best = np.inf
        self.best_model = copy.deepcopy(self.model)
        for i in range(self.est_params['max_its']):
            print(f'epoch {i+1}')
            self.train_loop()
            if self.validation_loop(i+1):
                break
    def model_eval(self,X,Z,X_q_test):
        if self.type in ['ones','random_uniform']:
            self.X_q_test = X_q_test
            if self.X_q_test.dim() == 1:
                self.X_q_test = self.X_q_test.unsqueeze(-1)
            return self.w
        n = X.shape[0]
        self.X_q_test = X_q_test
        if self.X_q_test.dim()==1:
            self.X_q_test = self.X_q_test.unsqueeze(-1)
        with torch.no_grad():
            if self.type == 'rulsif':
                w = self.model.get_w(X, Z, self.X_q_test)
            else:
                self.model = self.best_model
                w = self.model.get_w(X, Z,[])
        _w = w.cpu().squeeze().numpy()
        return w

    def model_eval_cat(self,X,Z,X_cont):
        with torch.no_grad():
            self.model_cat=self.best_model_cat
            w = self.model_cat.get_w(X, Z,X_cont)
        _w = w.cpu().squeeze().numpy()
        return w

    def return_weights(self,X,Z,X_Q):

        if self.type in ['random_uniform','ones']:
            return self.w
        else:
            self.w = 1.0
            if self.est_params['separate']:
                if self.has_cat:
                    w_cat= self.model_eval_cat(X[:,self.cat_x_marker],Z,X[:,~self.cat_x_marker])
                    self.w = self.w*w_cat.squeeze()
                if self.has_cont:
                    w_cont= self.model_eval(X[:,~self.cat_x_marker],Z,X_Q)
                    self.w = self.w * w_cont.squeeze()
            else:
                self.w  = self.model_eval(X,Z,X_Q)

            return self.w.squeeze()

    def get_median_ls_XY(self,X,Y):
        with torch.no_grad():
            if X.shape[0]>5000:
                X = X[torch.randperm(5000),:]
            if Y.shape[0]>5000:
                Y = Y[torch.randperm(5000),:]
            d = self.kernel_base.covar_dist(x1=X,x2=Y)
            ret = torch.sqrt(torch.median(d[d >= 0]))
            return ret

    def get_median_ls(self,X):
        with torch.no_grad():
            d = self.kernel_base.covar_dist(x1=X,x2=X)
            ret = torch.sqrt(torch.median(d[d >= 0]))
            return ret

    # def kernel_ls_init_keops(self,ls,data,data_2=None):
    #     with torch.no_grad():
    #         ker = keops_RBFkernel(ls=1/ls.unsqueeze(0),x=data,y=data_2,device_id=self.device)
    #         return ker()

    def kernel_ls_init(self,name,data,data_2=None,ls=None):
        ker = RBFKernel()
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

def permute(n_bins,og_indices,clusters):
    permutation = copy.deepcopy(og_indices)
    for i in range(n_bins):
        mask = i==clusters
        group = og_indices[mask]
        permuted_group=np.random.permutation(group)
        permutation[mask]=permuted_group
    return permutation

class Q_weighted_HSIC(): # test-statistic seems to be to sensitive???
    def __init__(self,X,Y,w,X_q,cuda=False,device=0,half_mode=False,perm='Y',variant=1,seed=1):
        # torch.random.manual_seed(seed)
        self.X = X
        self.Y = Y
        self.perm = perm
        self.X_q = X_q
        self.n = X.shape[0]
        self.w = w.unsqueeze(-1) #works better with the bug haha
        self.cuda = cuda
        self.device = device
        self.half_mode = half_mode
        self.variant = variant
        if self.X_q.dim()==1:
            self.X_q = self.X_q.unsqueeze(-1)

        with torch.no_grad():
            if self.variant==1:
                self.kernel_X_X_q = RBFKernel().cuda(device)
                _ls = self.get_median_ls(self.X,self.X_q) #This is critical, figure out way to get the right lr
                self.kernel_X_X_q._set_lengthscale(_ls)
            elif self.variant==2:
                self.kernel_X_X_q = LinearKernel().cuda(device)
                self.kernel_X_X_q._set_lengthscale(1.0)

            self.k_X_X_q = self.kernel_X_X_q(self.X, self.X_q).evaluate()
            for name, data in zip(['X', 'Y','X_q'], [self.X, self.Y,self.X_q]):
                self.kernel_ls_init(name, data)

            self.const_var_1_X = self.kernel_Y @ self.w
            self.const_var_1_Y = self.k_X_X_q.sum(dim=1, keepdim=True)
            self.const_sum_Y = self.kernel_X_q.sum()
            self.a_2 = self.const_sum_Y * (self.w.t() @ self.const_var_1_X) / self.n ** 4
            self.W = self.w @ self.w.t()

    def get_permuted2d(self,ker):
        idx = torch.randperm(self.n)
        kernel_X = ker[:,idx]
        kernel_X = kernel_X[idx,:]
        return kernel_X,idx

    def get_permuted1d(self,ker):
        idx = torch.randperm(self.n)
        kernel_X = ker[idx,:]
        return kernel_X,idx

    def calculate_weighted_statistic(self):
        with torch.no_grad():
            a_1 = self.w.t() @ (self.kernel_X * self.kernel_Y) @ self.w / self.n ** 2
            a_3 = self.w.t() @ (self.const_var_1_Y * (self.const_var_1_X)) / self.n ** 3
            return self.n*(a_1+self.a_2-2*a_3).squeeze()

    def permutation_calculate_weighted_statistic(self):
        with torch.no_grad():
            kernel_Y, idx = self.get_permuted2d(self.kernel_Y)
            a_1 = self.w.t() @ (self.kernel_X * kernel_Y) @ self.w / self.n ** 2
            a_3 = self.w.t() @ (self.const_var_1_Y * kernel_Y@self.w) / self.n ** 3
            a_2 = self.const_sum_Y * torch.sum(kernel_Y*self.W) / self.n ** 4
            return self.n * (a_1 + a_2 - 2 * a_3)

    def kernel_ls_init(self, name, data):
        if self.variant == 1:
            setattr(self, f'ker_obj_{name}',
                    RBFKernel().cuda(self.device) if self.cuda else RBFKernel())
            ls = self.get_median_ls(data)
            getattr(self, f'ker_obj_{name}')._set_lengthscale(ls)
            setattr(self, f'kernel_{name}', getattr(self, f'ker_obj_{name}')(data).evaluate())
        elif self.variant==2:
            setattr(self, f'ker_obj_{name}',
                    LinearKernel().cuda(self.device) if self.cuda else LinearKernel())
            getattr(self, f'ker_obj_{name}')._set_lengthscale(1.0)
            setattr(self, f'kernel_{name}', getattr(self, f'ker_obj_{name}')(data).evaluate())

    def get_median_ls(self, X,Y=None):
        kernel_base = Kernel().cuda(self.device)
        with torch.no_grad():
            if Y is None:
                d = kernel_base.covar_dist(x1=X, x2=X)
            else:
                d = kernel_base.covar_dist(x1=X, x2=Y)
            ret = torch.sqrt(torch.median(d[d >= 0])) # print this value, should be increasing with d
            if ret.item()==0:
                ret = torch.tensor(1.0)
            return ret


class Q_weighted_HSIC_correct(Q_weighted_HSIC): # test-statistic seems to be to sensitive???
    def __init__(self,X,Y,w,X_q,cuda=False,device=0,half_mode=False,perm='Y',variant=1,seed=1):
        super(Q_weighted_HSIC_correct, self).__init__(X,Y,w,X_q,cuda,device,half_mode,perm,variant,seed)
        self.og_indices=np.arange(X.shape[0])
        self.n_bins=X.shape[0]//20
        self.binner = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')
        numpy_w=w.cpu().numpy()
        self.clusters = self.binner.fit_transform(numpy_w[:,np.newaxis]).squeeze()

    def get_permuted2d(self,ker):
        idx = permute(self.n_bins,self.og_indices,self.clusters)
        kernel_X = ker[:,idx]
        kernel_X = kernel_X[idx,:]
        return kernel_X,idx

    def permutation_calculate_weighted_statistic(self):
        with torch.no_grad():
            kernel_Y, idx = self.get_permuted2d(self.kernel_Y)
            a_1 = self.w.t() @ (self.kernel_X * kernel_Y) @ self.w / self.n ** 2
            a_3 = self.w.t() @ (self.const_var_1_Y * kernel_Y@self.w) / self.n ** 3
            a_2 = self.const_sum_Y * torch.sum(kernel_Y*self.W) / self.n ** 4
            return self.n * (a_1 + a_2 - 2 * a_3)

class time_series_Q_hsic(Q_weighted_HSIC):
    def __init__(self,X,Y,w,X_q,cuda=False,device=0,half_mode=False,perm='Y',variant=1,within_perm_vec=None,n_blocks=20):
        super(time_series_Q_hsic, self).__init__(X,Y,w,X_q,cuda,device,half_mode,perm,variant,1)
        if within_perm_vec is not None: #shape is [a ,b] (chunk of chukns)
            self.within_block = True
            idx = torch.arange(X.shape[0])
            self.big_chunk_list = []
            for el in within_perm_vec:
                sub_vec = idx[el[0]:el[1]]
                sub_chunks = torch.chunk(sub_vec,n_blocks)
                self.big_chunk_list.append(sub_chunks)
        else:
            self.within_block = False
            self.n_blocks =  n_blocks
            self.chunked_indices = torch.chunk(torch.arange(X.shape[0]),self.n_blocks)
            self.len_chunked = len(self.chunked_indices)

    def within_block_permute(self):
        cat_indices = []
        for el in self.big_chunk_list:
            sampled_elements = random.sample(el, len(el))
            cat_indices.append(torch.cat(sampled_elements))
        idx = torch.cat(cat_indices)
        return idx

    def block_permute(self):
        sampled_elements = random.sample(self.chunked_indices,self.len_chunked)
        block_permute = torch.cat(sampled_elements)
        return block_permute

    def get_permuted2d(self,ker):
        if self.within_block:
            idx = self.within_block_permute()
        else:
            idx = self.block_permute()
        kernel_X = ker[:,idx]
        kernel_X = kernel_X[idx,:]
        return kernel_X,idx



