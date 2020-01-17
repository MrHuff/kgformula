import torch
from kgformula.utils import calculate_power,hypothesis_acceptance
import gpytorch
class weighted_stat():
    def __init__(self,X,Y,Z,do_null = True):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.do_null=do_null
        # self.dependence =

    def calculate_statistic(self):


        pass