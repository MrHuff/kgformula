import pandas as pd
import os
import torch
if __name__ == '__main__':
    dirname ='do_null_binary_csv'
    new_dirname='do_null_binary'
    if not os.path.exists(new_dirname):
        os.makedirs(new_dirname)
    files_to_process = os.listdir(dirname)
    for i,f in enumerate(files_to_process):
        df = pd.read_csv(dirname+'/'+f)
        dat = torch.from_numpy(df.values)
        Z = dat[:,0].unsqueeze(-1).float()
        Y=dat[:,-1].unsqueeze(-1).float()
        X = dat[:,1:3].float()
        torch.save((X,Y,Z,X),new_dirname+'/'+f'data_seed={i}.pt')


