import pandas as pd
import os
import torch
import numpy as np
if __name__ == '__main__':
    n = 10000
    seeds = 100
    for d in [1, 3, 50]:
        for null_case in [True, False]:
            for alp in [0.0, 2 * 1e-2, 4 * 1e-2, 6 * 1e-2, 8 * 1e-2, 1e-1]:
                new_dirname = f'do_null_univariate_alp={alp}_null={null_case}_d={d}'
                new_dirname_csv = f'do_null_univariate_alp={alp}_null={null_case}_d={d}_csv'
                if not os.path.exists(new_dirname_csv):
                    os.makedirs(new_dirname_csv)
                for i in range(seeds):
                    x,y,z,w = torch.load(new_dirname + '/' + f'data_seed={i}.pt')
                    x_csv = x.cpu().numpy()
                    y_csv = y.cpu().numpy()
                    z_csv = z.cpu().numpy()
                    np.savetxt(f"{new_dirname_csv}/x_{i}.csv", x_csv, delimiter=",")
                    np.savetxt(f"{new_dirname_csv}/y_{i}.csv", y_csv, delimiter=",")
                    np.savetxt(f"{new_dirname_csv}/z_{i}.csv", z_csv, delimiter=",")


