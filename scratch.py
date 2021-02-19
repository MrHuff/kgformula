import torch

from kgformula.utils import *
seeds=100
beta_xy=[0.1,0.0]
d_X=1
d_Y=1
d_Z=1
n=1000
yz=0.5
b_z=0.1
# b_z=0.5
b_z = (d_Z ** 2) * b_z
# theta=2.0
# phi=2.0
phi=1.25
theta=0.75
if __name__ == '__main__':
    data_dir = f"exponential_{seeds}/beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n={n}_yz={yz}_beta_XZ={round(b_z / (d_Z ** 2), 3)}_theta={theta}_phi={round(phi, 2)}"
    # data_dir = f"data_{seeds}/beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n={n}_yz={yz}_beta_XZ={round(b_z / (d_Z ** 2), 3)}_theta={theta}_phi={round(phi, 2)}"

    X,Y,Z,p_XZ = torch.load(data_dir+'/data_seed=0.pt') #Incorrect distribution!
    xq_class = x_q_class(qdist=2,q_fac=0.5,X=X) #Ok seems that support should be ok  now???
    xq_class.calc_w_q_sanity_exp(p_XZ)
    xq_class = x_q_class(qdist=2,q_fac=0.5,X=X) #Ok seems that support should be ok  now???
    xq_class.calc_w_q(p_XZ)

    # q = Exponential(rate=1)
    # z_ref = q.sample([Z.shape[0]])
    # cdf_z = q.cdf(Z)
    # range = torch.from_numpy(np.linspace(0,5,100))
    # pdf = torch.exp(q.log_prob(range))
    # plt.hist(Z.numpy(),bins=50)
    # plt.plot(range.numpy().squeeze(),pdf.numpy().squeeze())
    # plt.savefig('robin_debug.png')
    # stat,pval = kstest(cdf_z.numpy().squeeze(),'uniform')
    # print(stat,pval)
    # Rejection sampling allows you to magically sample a specific distribution without screwing up your marginals on the conditional distribution your trying to sample
    # Case in point p(x|z) retains x and z marginally, p(x,y,z) = p(z)*p(x|z) *p(y|x,z). If rejecting on p(x|z) the other components will not change
    # Marginalizing over different distribution results in different marginals.
    # load some data
    # plot histograms
    # try to construct a q distribution that's sensible
    #