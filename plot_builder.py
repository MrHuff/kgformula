
def build_suffix(q_fac,required_n,estimator):
    if estimator=='real_weights':
        estimate=False
        split_data=False
    else:
        estimate=True
        split_data=True
    suffix = f'_qf={q_fac}_qd=2_m=Q_s=0_1_e={estimate}_est={estimator}_sp={split_data}_br=250_n={required_n}'
    return suffix

def build_path(bxy,dz,bxz,theta,phi):
    if dz==1:
        list_xy = [0.0,bxy]
        dx=1
        dy=1
    else:
        list_xy = [0,bxy]
        dx=3
        dy=3
    path = f'beta_xy={list_xy}_d_X={dx}_d_Y={dy}_d_Z={dz}_n=10000_yz=0.5_beta_XZ={bxz}_theta={theta}_phi={phi}/layers=3_width=32/'
    return path
def build_histogram_plots(folder,beta_xy,):
    pass





if __name__ == '__main__':
    pass