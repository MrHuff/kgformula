from generate_data_multivariate import *
from kgformula.utils import *
import seaborn as sns
import sklearn

#TODO's fix the sanity checks to be general
#TODO make sure estimators work in 1-d for HSIC and GCM breakers
#TODO write up

def sanity_check_marginal_z(Z,fam_z):
    if fam_z==3:
        q = Exponential(rate=1)  # Bug in code you are not sampling exponentials!!!!!
    elif fam_z==1:
        q = Normal(loc=0, scale=1)  # This is still OK
    range = torch.from_numpy(np.linspace(0, 5, 100))
    cdf_z = q.cdf(Z)
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

def sanity_check_marginal_y(X,Y,beta_xy,fam_y):
    a = beta_xy[0]
    b = beta_xy[1]

    if fam_y==3:
        if torch.is_tensor(b):
            p = Exponential(rate=torch.exp(a + X @ b))  # Consider square matrix valued b.
        else:
            p = Exponential(rate=torch.exp(a + X * b))  #
    elif fam_y==1:
        if torch.is_tensor(b):
            p = Normal(loc=a+X@b,scale=1) #Consider square matrix valued b.
        else:
            p = Normal(loc=a+X*b,scale=1) #Consider square matrix valued b.

    cdf_y = p.cdf(Y)
    stat, pval = kstest(cdf_y.numpy().squeeze(), 'uniform')
    plt.hist(cdf_y.numpy(), bins=50,density=True)
    plt.savefig('Y_marg_histogram.png')
    plt.clf()
    print(f'pval Y-marginal:{pval}')

def conditional_dependence_plot_1d(X,Y,Z):
    discretizer = sklearn.preprocessing.KBinsDiscretizer(n_bins=100,strategy='uniform',encode='ordinal')
    test = discretizer.fit_transform(Z.numpy())
    mask = test==50
    x_subset = X[mask]
    y_subset = Y[mask]
    plt.scatter(x_subset.numpy(),y_subset.numpy())
    plt.savefig('conditional_plot.png')

def pairs_plot(X,Y,Z):
    data = torch.cat([X,Y,Z],dim=1).numpy()
    df = pd.DataFrame(data,columns=['X','Y','Z'])
    plot= sns.pairplot(df)
    plot.savefig('pairs_plot_1d.png')
    plt.clf()

def gen_data_and_sanity(data_dir,n,d_Z,beta_xz,beta_xy,i,yz,d_X,d_Y,phi,theta,fam_x,fam_z,fam_y):
    X, Y, Z, inv_w = simulate_xyz_multivariate(n, oversamp=100, d_Z=d_Z, beta_xz=beta_xz,
                                               beta_xy=beta_xy, seed=i, yz=yz, d_X=d_X, d_Y=d_Y,
                                               phi=phi, theta=theta, fam_x=fam_x, fam_z=fam_z, fam_y=fam_y)
    beta_xz = torch.Tensor(beta_xz)
    if i == 0:  # No way to sanity check...

        if d_X == 1:
            sanity_check_marginal_z(Z, fam_z=fam_z)
            sanity_check_marginal_y(X, Y, beta_xy, fam_y=fam_y)
            pairs_plot(X, Y, Z)
            conditional_dependence_plot_1d(X, Y, Z)
        _x_mu = torch.cat([torch.ones(*(X.shape[0], 1)), Z],
                          dim=1) @ beta_xz  # XZ dependence
        mu = torch.exp(_x_mu + theta)
        for d in range(d_X):
            plt.hist(X[:, d].squeeze().numpy(), bins=100)
            plt.savefig('marginal_X.png')
            plt.clf()

            if fam_z == 3:
                test_d = torch.distributions.Gamma(rate=1.0, concentration=theta)
                sample = test_d.sample([n])
                test = X[:, d].squeeze() / mu
                stat, pval = kstest(test.squeeze().numpy(), sample.squeeze().numpy())
                sample = sample.numpy()
            elif fam_z == 1:
                test = torch.cat([torch.ones(*(X.shape[0], 1)), Z],
                                 dim=1) @ beta_xz  # XZ dependence
                sig_xxz = theta
                sample = (X[:, d] - test).squeeze().numpy()  # *sig_xxz
                stat, pval = kstest(sample, 'norm', (0, sig_xxz))
                print(f'KS gamma test pval {pval}')

            plt.hist(test.squeeze().numpy(), bins=100, color='b', alpha=0.25)
            plt.hist(sample.squeeze(), bins=100, color='r', alpha=0.25)
            plt.savefig('test_vs_dist.png')
            plt.clf()

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

def multiprocess_wrapper(args):
    gen_data_and_sanity(*args)

if __name__ == '__main__': #Replace GCM with something else that works/has power
    seeds = 1
    yz = [-0.5,1.0] #GCM breaker - coplua
    xy_const = 0.0 #GCM breaker
    fam_z = 1
    fam_y= 1
    fam_x =[1,1]
    folder_name = f'exp_gcm_break_{seeds}'
    jobs = []
    for d_X,d_Y,d_Z, theta,phi in zip( [1],[1],[1],[1.0],[2.0]): #GCM BREAKER
        for b_z in [0.0]:  # ,1e-3,1e-2,0.05,0.1,0.25,0.5,1
            b_z = (d_Z ** 2) * b_z
            beta_xz = generate_sensible_variables(d_Z, b_z, const=0)  # What if X and Z indepent -> should be uniform, should sanity check that this actully is well behaved for all d_Z.
            for n in [10000]:
                # for beta_xy in [[xy_const, 0.0],[xy_const, 0.5]]:
                for beta_xy in [[xy_const, 0.0]]:
                    data_dir = f"{folder_name}/beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n={n}_yz={yz}_beta_XZ={round(b_z / (d_Z ** 2), 3)}_theta={theta}_phi={round(phi, 2)}"
                    if not os.path.exists(f'./{data_dir}/'):
                        os.makedirs(f'./{data_dir}/')
                    for i in range(seeds):
                        jobs.append([data_dir,n,d_Z,beta_xz,beta_xy,i,yz,d_X,d_Y,phi,theta,fam_x,fam_z,fam_y])
    # import torch.multiprocessing as mp
    # pool = mp.Pool(mp.cpu_count())
    # pool.map(multiprocess_wrapper, [row for row in jobs])
    multiprocess_wrapper(jobs[0])