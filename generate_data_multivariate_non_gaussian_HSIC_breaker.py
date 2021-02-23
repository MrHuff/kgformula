from generate_data_multivariate import *
from kgformula.utils import *
def sanity_check_marginal_z(Z):
    q = Exponential(rate=1)  # Bug in code you are not sampling exponentials!!!!!
    cdf_z = q.cdf(Z)
    range = torch.from_numpy(np.linspace(0, 5, 100))
    pdf = torch.exp(q.log_prob(range))
    plt.hist(Z.numpy(), bins=50,density=True)
    plt.plot(range.numpy().squeeze(), pdf.numpy().squeeze())
    plt.savefig('Z_debug.png')
    plt.clf()
    stat, pval = kstest(cdf_z.numpy().squeeze(), 'uniform')
    plt.hist(cdf_z.numpy(), bins=50,density=True)
    plt.savefig('Z_marg_histogram.png')
    plt.clf()
    print(f'pval z-marginal:{pval}')

def sanity_check_marginal_y(X,Y,beta_xy):
    a = beta_xy[0]
    b = beta_xy[1]

    if torch.is_tensor(b):
        p = Exponential(rate=torch.exp(a + X @ b))  # Consider square matrix valued b.
    else:
        p = Exponential(rate=torch.exp(a + X * b))  #
    cdf_y = p.cdf(Y)
    stat, pval = kstest(cdf_y.numpy().squeeze(), 'uniform')
    plt.hist(cdf_y.numpy(), bins=50,density=True)
    plt.savefig('Y_marg_histogram.png')
    plt.clf()
    print(f'pval Y-marginal:{pval}')

if __name__ == '__main__':
    seeds = 100
    yz = [0.5,0.0]
    xy_const = 0.25

    for d_X,d_Y,d_Z, theta,phi in zip( [1],[1],[1],[0.1],[0.9]): #GCM BREAKER
        for b_z in [0.5]:  # ,1e-3,1e-2,0.05,0.1,0.25,0.5,1
            b_z = (d_Z ** 2) * b_z
            beta_xz = generate_sensible_variables(d_Z, b_z, const=0)  # What if X and Z indepent -> should be uniform, should sanity check that this actully is well behaved for all d_Z.
            # Try different beta configs, i.e.
            # Smaller theta, but not too small. Acceptable SNR...
            # Somewhere in between for d_Z
            # High SNR for X to Z makes it harder
            # "perfect" balance between X and Z make it just hard enough so TRE and NCE_Q can get the job done.
            # for n in [1000,5000,10000]:

            for n in [1000]:
                # for beta_xy in [[0, 0.5],[0, 0.0], [0, 0.25], [0, 0.1]]:
                # for beta_xy in [[0, 0.5],[0, 0.0]]:
                for beta_xy in [[xy_const, 0.0],[xy_const, 0.5]]:
                    data_dir = f"exp_hsic_break_{seeds}/beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n={n}_yz={yz}_beta_XZ={round(b_z / (d_Z ** 2), 3)}_theta={theta}_phi={round(phi, 2)}"
                    if not os.path.exists(f'./{data_dir}/'):
                        os.makedirs(f'./{data_dir}/')
                    for i in range(seeds):
                        if not os.path.exists(f'./{data_dir}/data_seed={i}.pt'):
                            X, Y, Z, inv_w = simulate_xyz_multivariate(n, oversamp=5, d_Z=d_Z, beta_xz=beta_xz,
                                                                       beta_xy=beta_xy, seed=i, yz=yz, d_X=d_X, d_Y=d_Y,
                                                                       phi=phi, theta=theta,fam_x=[3,3],fam_z=3,fam_y=2)
                            beta_xz = torch.Tensor(beta_xz)
                            if i == 0: #No way to sanity check...
                                if d_X==1:
                                    sanity_check_marginal_z(Z)
                                    sanity_check_marginal_y(X, Y, beta_xy)

                                _x_mu = torch.cat([torch.ones(*(X.shape[0], 1)), Z],
                                                 dim=1) @ beta_xz  # XZ dependence
                                mu = torch.exp(_x_mu+theta)
                                for d in range(d_X):

                                    plt.hist(X[:,d].squeeze().numpy(),bins=100)
                                    plt.savefig('test_X.png')
                                    plt.clf()
                                    test_d = torch.distributions.Gamma(rate= 1.0, concentration=theta)
                                    sample = test_d.sample([n])
                                    test = X[:,d].squeeze()/mu
                                    plt.hist(test.squeeze().numpy(),bins=100,color='b',alpha=0.25)
                                    plt.hist(sample.squeeze().numpy(), bins=100,color='r',alpha=0.25)
                                    plt.savefig('test_vs_dist.png')
                                    plt.clf()
                                    stat, pval = kstest(test.squeeze().numpy(), sample.squeeze().numpy())
                                    print(f'KS gamma test pval {pval}')
                                p_val = hsic_test(X[0:1000, :], Z[0:1000, :], 250)
                                X_class = x_q_class(qdist=2, q_fac=1.0, X=X)
                                w = X_class.calc_w_q(inv_w)
                                sanity_pval = hsic_sanity_check_w(w[0:1000], X[0:1000, :], Z[0:1000, :], 250)
                                print(f'HSIC X Z: {p_val}')
                                print(f'sanity_check_w : {sanity_pval}')
                                plt.hist(w, bins=250)
                                plt.savefig(f'./{data_dir}/w.png')
                                plt.clf()
                                ess_list = []
                                for q in [1.0, 0.75, 0.5]:
                                    X_class = x_q_class(qdist=2, q_fac=q, X=X)
                                    w = X_class.calc_w_q(inv_w)
                                    ess = calc_ess(w)
                                    ess_list.append(ess.item())
                                max_ess = max(ess_list)
                                print('max_ess: ', max_ess)
                            torch.save((X, Y, Z, inv_w), f'./{data_dir}/data_seed={i}.pt')

