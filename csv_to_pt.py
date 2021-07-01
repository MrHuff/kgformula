import pandas as pd
import os
import torch
if __name__ == '__main__':
    dirname ='do_null_binary_csv_debug'
    new_dirname='do_null_binary_debug'
    if not os.path.exists(new_dirname):
        os.makedirs(new_dirname)
    files_to_process = os.listdir(dirname)
    for i,f in enumerate(files_to_process):
        df = pd.read_csv(dirname+'/'+f)
        dat = torch.from_numpy(df.values)
        Z = dat[:,0].unsqueeze(-1).float()
        Y=dat[:,2].unsqueeze(-1).float()
        X = dat[:,1].unsqueeze(-1).float()
        w = dat[:,-1].float()
        torch.save((X,Y,Z,w),new_dirname+'/'+f'data_seed={i}.pt')


