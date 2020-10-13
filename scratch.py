import matplotlib.pyplot as plt
import numpy as np
from kgformula.utils import torch_to_csv

n=10000
seeds=1
ground_truth = 'H_0'
theta = 2.0
phi = 2.0
for n in [10000]:
    for y_a in [0.0]:
        for y_b in [0.0]:
            for z_a in [0.0]:
                for z_b in [0.0, 1e-2, 0.1, 0.5, 1]:
                    for cor in [0.5]:
                        for q_fac in [0.4, 0.2, 0.01, 0.6, 0.8, 0.9, 1.0]:
                                data_dir = f'univariate_{seeds}_seeds/Q={q_fac}_gt={ground_truth}_y_a={y_a}_y_b={y_b}_z_a={z_a}_z_b={z_b}_cor={cor}_n={n}_seeds={seeds}_{theta}_{round(phi,2)}'
                                torch_to_csv(data_dir+'/','data_seed=0.pt')


# # generate 2 2d grids for the x & y bounds
# #-10,10 etc calculate all values plot. do the same for "predictions..."
# y, x = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
#
# z = (1 - x / 2. + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
# print(x.shape)
# print(y.shape)
# print(z)
# # x and y are bounds, so z should be the value *inside* those bounds.
# # Therefore, remove the last value from the z array.
# z = z[:-1, :-1]
# z_min, z_max = -np.abs(z).max(), np.abs(z).max()
#
# fig, ax = plt.subplots()
#
# c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
# ax.set_title('pcolormesh')
# # set the limits of the plot to the limits of the data
# ax.axis([x.min(), x.max(), y.min(), y.max()])
# fig.colorbar(c, ax=ax)
#
# plt.show()