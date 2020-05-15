import torch
import os

if __name__ == '__main__':
    file_name ='beta_xy=[0, 0.5]_d_X=3_d_Y=3_d_Z=50_n=1000_yz=0.5_beta_XZ=[0.0, 0.25, 0.25, 0.25, 0.25]'
    new_file_name = 'beta_xy=[0, 0.5]_d_X=3_d_Y=3_d_Z=50_n=10000_yz=0.5_beta_XZ=[0.0, 0.25, 0.25, 0.25, 0.25]'
    file_num  = 1000
    cat_rate = 10
    X_tmp  = []
    Y_tmp  = []
    Z_tmp  = []
    w_tmp = []
    if not os.path.exists(new_file_name):
        os.makedirs(new_file_name)
    j=0
    for i in range(file_num):
        x,y,z,w = torch.load(file_name+f'/data_seed={i}.pt')
        X_tmp.append(x)
        Y_tmp.append(y)
        Z_tmp.append(z)
        w_tmp.append(w)
        if (i+1)%cat_rate==0:
            X_cat = torch.cat(X_tmp,dim=0)
            Y_cat = torch.cat(Y_tmp,dim=0)
            Z_cat = torch.cat(Z_tmp,dim=0)
            w_cat = torch.cat(w_tmp,dim=0)
            print(new_file_name+f'/data_seed={j}.pt')
            torch.save((X_cat,Y_cat,Z_cat,w_cat),new_file_name+f'/data_seed={j}.pt')
            X_tmp = []
            Y_tmp = []
            Z_tmp = []
            w_tmp = []
            j+=1


