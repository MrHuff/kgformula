import torch

from kgformula.utils import *
seeds=100
beta_xy=[0,0.0]
d_X=1
d_Y=1
d_Z=1
n=5000
yz=0.5
b_z=0.1
b_z = (d_Z ** 2) * b_z
theta=1.0
phi=1.0
if __name__ == '__main__':
    data_dir = f"exponential_{seeds}/beta_xy={beta_xy}_d_X={d_X}_d_Y={d_Y}_d_Z={d_Z}_n={n}_yz={yz}_beta_XZ={round(b_z / (d_Z ** 2), 3)}_theta={theta}_phi={round(phi, 2)}"
    X,Y,Z,p_XZ = torch.load(data_dir+'/data_seed=0.pt')
    xq_class = x_q_class(qdist=2,q_fac=0.5,X=X) #Ok seems that support should be ok  now???
    xq_class.calc_w_q_sanity_exp(p_XZ)
    xq_class.calc_w_q(p_XZ)




    # load some data
    # plot histograms
    # try to construct a q distribution that's sensible
    #