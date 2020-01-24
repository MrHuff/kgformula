from kgformula.test_statistics import weighted_stat
from kgformula.do_samplers import do_distribution
import os
import torch
if __name__ == '__main__':

    if not os.path.exists('./simluated_do_null/'):
        X, Y, Z, w = do_distribution(n=100, get_p_x_cond_z=True)
        os.mkdir('./simluated_do_null/')
        torch.save((X,Y,Z,w),'./simluated_do_null/data.pt')
    else:
        X,Y,Z,w = torch.load('./simluated_do_null/data.pt')
        # X = X.unsqueeze(-1)
        # Y = Y.unsqueeze(-1)
        # Z = Z.unsqueeze(-1)

    #Cheating case
    c = weighted_stat(X=X,Y=Y,Z=Z,get_p_x_cond_z=w)