import pandas as pd
import os
import torch
import numpy as np
if __name__ == '__main__':
    # new_dirname = f'exp_gcm_break_100/beta_xy=[0.0, 0.0]_d_X=1_d_Y=1_d_Z=1_n=10000_yz=[-0.5, 4.0]_beta_XZ=0.0_theta=1.0_phi=2.0'
    # new_dirname_csv = f'exp_gcm_break_100/beta_xy=[0.0, 0.0]_d_X=1_d_Y=1_d_Z=1_n=10000_yz=[-0.5, 4.0]_beta_XZ=0.0_theta=1.0_phi=2.0_csv/'
    # seeds=100
    # if not os.path.exists(new_dirname_csv):
    #     os.makedirs(new_dirname_csv)
    # for i in range(seeds):
    #     x,y,z,w = torch.load(new_dirname + '/' + f'data_seed={i}.pt')
    #     x_csv = x.cpu().numpy()
    #     y_csv = y.cpu().numpy()
    #     z_csv = z.cpu().numpy()
    #     np.savetxt(f"{new_dirname_csv}/x_{i}.csv", x_csv, delimiter=",")
    #     np.savetxt(f"{new_dirname_csv}/y_{i}.csv", y_csv, delimiter=",")
    #     np.savetxt(f"{new_dirname_csv}/z_{i}.csv", z_csv, delimiter=",")

    n = 10000
    seeds = 100
    for d in [1]:
        for null_case in [False]:
            for alp in [0.0,0.005,0.01,2*1e-2,4*1e-2,6*1e-2,8*1e-2,1e-1]:
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
    # for beta_xy in [[0, 0.0], [0, 0.01], [0, 0.02], [0, 0.03], [0, 0.04], [0, 0.05]]:
    #
    #     new_dirname = f'do_null_100/beta_xy={beta_xy}_d_X=1_d_Y=1_d_Z=1_n=10000_yz=[0.5, 0.0]_beta_XZ=0.75_theta=2.0_phi=2.0'
    #     new_dirname_csv = f'do_null_100/beta_xy={beta_xy}_d_X=1_d_Y=1_d_Z=1_n=10000_yz=[0.5, 0.0]_beta_XZ=0.75_theta=2.0_phi=2.0'
    #     seeds=100
    #     if not os.path.exists(new_dirname_csv):
    #         os.makedirs(new_dirname_csv)
    #     for i in range(seeds):
    #         x,y,z,w = torch.load(new_dirname + '/' + f'data_seed={i}.pt')
    #         x_csv = x.cpu().numpy()
    #         y_csv = y.cpu().numpy()
    #         z_csv = z.cpu().numpy()
    #         np.savetxt(f"{new_dirname_csv}/x_{i}.csv", x_csv, delimiter=",")
    #         np.savetxt(f"{new_dirname_csv}/y_{i}.csv", y_csv, delimiter=",")
    #         np.savetxt(f"{new_dirname_csv}/z_{i}.csv", z_csv, delimiter=",")

    # for beta_xy in [[0, 0.0], [0, 0.01], [0, 0.02],[0, 0.03],[0, 0.04],[0, 0.05]]:
    #
    #     new_dirname = f'hdm_breaker_fam_y=4_100/beta_xy={beta_xy}_d_X=1_d_Y=1_d_Z=1_n=10000_yz=[0.5, 0.0]_beta_XZ=0.25_theta=2.0_phi=2.0'
    #     new_dirname_csv = f'hdm_breaker_fam_y=4_100_csv/beta_xy={beta_xy}_d_X=1_d_Y=1_d_Z=1_n=10000_yz=[0.5, 0.0]_beta_XZ=0.25_theta=2.0_phi=2.0'
    #     seeds=100
    #     if not os.path.exists(new_dirname_csv):
    #         os.makedirs(new_dirname_csv)
    #     for i in range(seeds):
    #         x,y,z,w = torch.load(new_dirname + '/' + f'data_seed={i}.pt')
    #         x_csv = x.cpu().numpy()
    #         y_csv = y.cpu().numpy()
    #         z_csv = z.cpu().numpy()
    #         np.savetxt(f"{new_dirname_csv}/x_{i}.csv", x_csv, delimiter=",")
    #         np.savetxt(f"{new_dirname_csv}/y_{i}.csv", y_csv, delimiter=",")
    #         np.savetxt(f"{new_dirname_csv}/z_{i}.csv", z_csv, delimiter=",")
    for beta_xy in [[0,0.0],[0,0.001],[0,0.002],[0,0.003],[0,0.004],[0,0.005]]:
        new_dirname = f'hdm_breaker_fam_y=4_100/beta_xy={beta_xy}_d_X=3_d_Y=3_d_Z=50_n=10000_yz=[0.5, 0.0]_beta_XZ=0.075_theta=16.0_phi=2.0'
        new_dirname_csv = f'hdm_breaker_fam_y=4_100_csv/beta_xy={beta_xy}_d_X=3_d_Y=3_d_Z=50_n=10000_yz=[0.5, 0.0]_beta_XZ=0.075_theta=16.0_phi=2.0'
        seeds=100
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