from kgformula.utils import *
def calc_ess(w):
    return (w.sum()**2)/(w**2).sum()
class simulation_object_rule_dummy():
    def __init__(self):
        self.device = 'cuda:0'
    def get_q_fac(self, X_org, Z_org):
        if X_org.shape[1] == 1 and Z_org.shape[1] == 1:
            X = standardize_variance(X_org)
            Z = standardize_variance(Z_org)
            cor = one_dim_corr(X, Z)
            # cor_output= kendalltau(X_org.cpu().squeeze(),Z_org.cpu().squeeze())
            # (cor,_)= spearmanr(X_org.cpu().squeeze(),Z_org.cpu().squeeze())
            return torch.sqrt(torch.clip((1 - 2 * cor.float()),
                                         1e-6))  # Analytical choice doesn't fucking work for exponential data... OK sigh, the normal business might not be doing a good job
        else:
            n = X_org.shape[0]
            X = standardize_variance(X_org)
            Z = standardize_variance(Z_org)
            I_p = torch.diag(X.var(0)).to(self.device)
            I_q = torch.diag(Z.var(0)).to(self.device)
            center_X = X
            center_Z = Z
            sigma_xz = 1 / n * center_X.t() @ center_Z
            sigma_xz = sigma_xz.to(self.device)
            inv_comp = torch.inverse(I_p - sigma_xz @ sigma_xz.t())
            B = inv_comp @ sigma_xz
            D = I_q - sigma_xz.t() @ (inv_comp @ sigma_xz)
            # Issue found D, should be PSD hence so det cant be negative...
            # det_const = torch.det(D)
            solve, _ = torch.solve(B.t(), D)  # Fix covariance estimation...
            subtract_term = inv_comp + B @ solve
            # c_q =torch.tensor(1.0).float().to(self.device)
            # c_q.requires_grad=True
            # opt = torch.optim.Adam(params=[c_q],lr=1e-2)
            # best = 1e99
            c_q_list = []
            losses = []
            its = 250
            # hmm keep it within 0 and 1?
            for i in range(its):
                c_q = 0.0+ 1.0 * (i) / its
                T_inv = torch.diag(1. / (torch.diag(I_p) * c_q))
                T = I_p * c_q
                loss = (1 / torch.det(T)) * (1 / torch.det(2 * T_inv - subtract_term) ** 0.5)
                if not torch.isnan(loss):
                    c_q_list.append(c_q)
                    losses.append(loss.item())
            idx_best = np.argmin(losses)
            best_c_q = c_q_list[idx_best]
            print('best c_q\n')
            print(best_c_q, losses[idx_best])
            # plt.plot(c_q_list,losses)
            # plt.savefig('plt_det_obj.png')

            return best_c_q
    def get_binary_mask(self,X):
        dim = X.shape[1]
        mask_ls = [0]*dim
        for i in range(dim):
            x = X[:,i]
            un_el = x.unique()
            mask_ls[i] = un_el.numel()==2
        return torch.tensor(mask_ls)

    def run(self,data_dir):
        q_fac_list = []
        required_n=10000
        self.qdist=2
        X, Y, Z, _w = torch.load(f'{data_dir}/data_seed=0.pt')
        X, Y, Z, _w = X.cuda(self.device), Y.cuda(self.device), Z.cuda(self.device), _w.cuda(self.device)
        X, Y, Z, _w = X[:required_n, :], Y[:required_n, :], Z[:required_n, :], _w[:required_n]
        n_half = X.shape[0] // 2
        X_train, X_test = split(X, n_half)
        Z_train, Z_test = split(Z, n_half)
        binary_mask_X = self.get_binary_mask(X)

        binary_mask_Z = self.get_binary_mask(Z)
        X_cont = X[:, ~binary_mask_X]
        X_bin = X[:, binary_mask_X]
        concat_q = []
        if X_bin.numel()>0:
            Xq_class_bin = x_q_class_bin(X=X_bin)
            X_q_bin = Xq_class_bin.sample(n=X_bin.shape[0])
            X_q_bin = X_q_bin.to(self.device)
            concat_q.append(X_q_bin)
        if X_cont.numel()>0:
            q_fac = self.get_q_fac(X_train[:, ~binary_mask_X], Z_train[:, ~binary_mask_Z])
            # q_fac = 1.0
            print('q_fac: \n ',q_fac)
            q_fac_list.append(q_fac)
            Xq_class_cont = x_q_class_cont(qdist=self.qdist, q_fac=q_fac, X=X_cont)
            w_q = Xq_class_cont.calc_w_q(_w)
            print(calc_ess(w_q))
if __name__ == '__main__':
    c = simulation_object_rule_dummy()
    for bxz in [0.0,0.05,0.1,0.15,0.2,0.25]:
        c.run(f'kchsic_breaker/beta_xy=[0.0, 0.0]_d_X=8_d_Y=8_d_Z=50_n=10000_yz=[0.5, 0.0]_beta_XZ={bxz}_theta=16.0_phi=2.0')
