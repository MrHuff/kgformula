from generate_data_multivariate import *

if __name__ == '__main__':
    seeds = 100
    yz = 0.5
    # for d_X,d_Y,d_Z, theta,phi in zip( [1,3,3,3],[1,3,3,3],[1,3,15,50],[1.,1.,1.,1.],[1.,1.,1.,1.]): #50,3
    for d_X,d_Y,d_Z, theta,phi in zip( [1],[1],[1],[1.],[1.]): #50,3
    # for d_X,d_Y,d_Z, theta,phi in zip( [3,3,3],[3,3,3],[3,15,50],[0.5,1,1],[2.0,2.0,2.0]): #50,3
    # for d_X,d_Y,d_Z, theta,phi in zip( [3],[3],[50],[1.],[1.]): #50,3 #Ok problem... Let's ask Robin a bit later...
    # for d_X, d_Y, d_Z, theta, phi in zip([3], [3], [3], [3.0], [2.0]):  # 50,3
    # for d_X, d_Y, d_Z, theta, phi in zip([1], [1], [1], [1.], [2.]):  # 50,3
    #     for b_z in [0.0,0.01,0.1,0.2,0.3,0.4,0.5]:  # ,1e-3,1e-2,0.05,0.1,0.25,0.5,1
        for b_z in [0.0,0.01,0.1,0.5]:  # ,1e-3,1e-2,0.05,0.1,0.25,0.5,1
            b_z = (d_Z ** 2) * b_z
            beta_xz = generate_sensible_variables(d_Z, b_z, const=0)  # What if X and Z indepent -> should be uniform, should sanity check that this actully is well behaved for all d_Z.
            # Try different beta configs, i.e.
            # Smaller theta, but not too small. Acceptable SNR...
            # Somewhere in between for d_Z
            # High SNR for X to Z makes it harder
            # "perfect" balance between X and Z make it just hard enough so TRE and NCE_Q can get the job done.
            for n in [1000,5000,10000]:
                # for beta_xy in [[0, 0.5],[0, 0.0], [0, 0.25], [0, 0.1]]:
                for beta_xy in [[0, 0.5],[0, 0.0]]:
                    data_dir = f"exponential_{seeds}/beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n={n}_yz={yz}_beta_XZ={round(b_z / (d_Z ** 2), 3)}_theta={theta}_phi={round(phi, 2)}"
                    if not os.path.exists(f'./{data_dir}/'):
                        os.makedirs(f'./{data_dir}/')
                    for i in range(seeds):
                        if not os.path.exists(f'./{data_dir}/data_seed={i}.pt'):
                            X, Y, Z, inv_w = simulate_xyz_multivariate(n, oversamp=5, d_Z=d_Z, beta_xz=beta_xz,
                                                                       beta_xy=beta_xy, seed=i, yz=yz, d_X=d_X, d_Y=d_Y,
                                                                       phi=phi, theta=theta,fam_x=[3,3],fam_z=1,fam_y=2,copula_fam=3)
                            beta_xz = torch.Tensor(beta_xz)
                            if i == 0: #No way to sanity check...
                                beta_xz = torch.tensor(beta_xz).float()
                                _x_mu = torch.cat([torch.ones(*(X.shape[0], 1)), Z],
                                                 dim=1) @ beta_xz  # XZ dependence
                                mu = torch.exp(_x_mu+theta)  # Poisson link func?
                                for d in range(d_X):
                                    test = X[:,d].squeeze()*(mu)
                                    plt.hist(test.squeeze().numpy(),bins=100)
                                    plt.savefig('test.png')
                                    plt.clf()
                                    plt.hist(X[:,d].squeeze().numpy(),bins=100)
                                    plt.savefig('test_X.png')
                                    plt.clf()
                                    test_d = torch.distributions.Gamma(rate= 0.5, concentration=1.)
                                    sample = test_d.sample([n])
                                    plt.hist(sample.squeeze().numpy(), bins=100)
                                    plt.savefig('sample_sanity.png')
                                    plt.clf()
                                    # stat, pval = kstest(test.squeeze().numpy(), sample.squeeze().numpy())
                                    # print(f'KS gamma test pval {pval}')
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

