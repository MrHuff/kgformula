import os
import GPUtil
try:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Get the first available GPU
    DEVICE_ID_LIST = GPUtil.getAvailable(order='memory', limit=1)
    DEVICE_ID = DEVICE_ID_LIST[0] # grab first element from list
    # Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
except:
    print('no gpu rip')
from kgformula.utils import split,x_q_class_bin,density_estimator
import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use( 'tkagg' )
def get_binary_mask(X):
    dim = X.shape[1]
    mask_ls = [0] * dim
    for i in range(dim):
        x = X[:, i]
        un_el = x.unique()
        mask_ls[i] = un_el.numel() <= 10
    return torch.tensor(mask_ls)
if __name__ == '__main__':
    estimator = 'NCE_Q'
    i=0
    data_dir = 'do_null_mix_100/beta_xy=[0, 0.0]_d_X=3_d_Y=3_d_Z=3_n=10000_yz=[0.5, 0.0]_beta_XZ=0.5_theta=4.0_phi=2.0'
    est_params= {'lr': 1e-3, #use really small LR for TRE. Ok what the fuck is going on...
                                                   'max_its': 10,
                                                   'width': 32,
                                                   'layers':4,
                                                   'bs_ratio': 1e-2,
                                                   'val_rate': 0.1,
                                                   'n_sample': 250,
                                                   'criteria_limit': 0.05,
                                                   'kill_counter': 2,
                                                    'kappa': 10 if estimator in ['NCE', 'NCE_Q'] else 1, #might need to adjust this for binary data?
                                                   'm': 4,
                                                    'separate':False
                                                   }
    device = 'cuda:0'
    X, Y, Z, _w = torch.load(f'./{data_dir}/data_seed={i}.pt')

    X, Y, Z, _w = X.cuda(device), Y.cuda(device), Z.cuda(device), _w.cuda(device)

    n_half = X.shape[0] // 2
    X_train, X_test = split(X, n_half)
    Y_train, Y_test = split(Y, n_half)
    Z_train, Z_test = split(Z, n_half)
    binary_mask_X = get_binary_mask(X)
    _, _w_plot = split(_w, n_half)

    _w_plot = _w_plot.cpu().numpy()
    _w_plot_plot = _w_plot[_w_plot<2]
    plt.hist(_w_plot_plot,bins=100)
    plt.show()
    plt.clf()
    print(_w_plot)
    X_cont = X[:, ~binary_mask_X]
    X_bin = X[:, binary_mask_X]
    Xq_class_bin = x_q_class_bin(X=X_bin)
    X_q_bin = Xq_class_bin.sample(n=X_bin.shape[0])
    concat_q=[]
    concat_q.append(X_q_bin)
    X_q = torch.cat(concat_q, dim=1)
    X_q = X_q.to(device)
    X_q_train, X_q_test = split(X_q, n_half)

    d = density_estimator(x=X_train, z=Z_train, x_q=X_q_train, cuda=True,
                          est_params=est_params, type=estimator, device=device,
                          secret_indx=9999)
    w = d.return_weights(X_test, Z_test, X_q_test)
    save_w = w.cpu().numpy()
    print(save_w)
    plot_save_w =  save_w[save_w<2]
    plt.hist(plot_save_w,bins=100)
    plt.show()
    plt.clf()
    err = np.abs(_w_plot-save_w)
    plt.hist(err,bins=100)
    plt.show()
    plt.clf()
