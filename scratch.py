from matplotlib import pyplot as plt
import torch

x,y,z,w=torch.load("do_null_univariate_alp=0.1_null=True/data_seed=0.pt")

x=x.numpy()
z=z.numpy()

bool_1 = x==0

z_slice = z[bool_1]
print(z_slice.mean())
plt.hist(z_slice,100)
plt.show()