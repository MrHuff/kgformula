import os
import shutil

def build_suffix(q_fac,required_n,estimator,br):
    if estimator=='real_weights':
        estimate=False
        split_data=False
    else:
        estimate=True
        split_data=True
    suffix = f'__qf={q_fac}_qd=2_m=Q_s=0_100_e={estimate}_est={estimator}_sp={split_data}_br={br}_n={required_n}'
    return suffix

def build_suffix_2(q_fac,required_n,estimator,br):
    if estimator=='real_weights':
        estimate=False
        split_data=False
    else:
        estimate=True
        split_data=True
    suffix = f'qf=rule_qd=2_m=Q_s=0_100_e={estimate}_est={estimator}_sp={split_data}_br={br}_n={required_n}'
    return suffix

def build_hsic(required_n,br):
    suffix = f'__hsic_s={0}_{100}_br={br}_n={required_n}'
    return suffix

def build_regression(required_n):
    suffix = f'__linear_reg={0}_{100}_n={required_n}'
    return suffix


def build_path(list_xy,dx,dy,dz,bxz,theta,phi,yz):
    path = f'beta_xy={list_xy}_d_X={dx}_d_Y={dy}_d_Z={dz}_n=10000_yz={yz}_beta_XZ={bxz}_theta={theta}_phi={phi}/'
    return path


