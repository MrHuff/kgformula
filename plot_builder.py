import os
import shutil

def build_suffix(q_fac,required_n,estimator):
    if estimator=='real_weights':
        estimate=False
        split_data=False
    else:
        estimate=True
        split_data=True
    suffix = f'__qf={q_fac}_qd=2_m=Q_s=0_100_e={estimate}_est={estimator}_sp={split_data}_br=250_n={required_n}'
    return suffix

def build_path(bxy,dz,bxz,theta,phi):
    if dz==1:
        list_xy = [0.0,int(bxy)]
        dx=1
        dy=1
    else:
        list_xy = [0,float(bxy)]
        dx=3
        dy=3
    path = f'beta_xy={list_xy}_d_X={dx}_d_Y={dy}_d_Z={dz}_n=10000_yz=0.5_beta_XZ={bxz}_theta={theta}_phi={phi}/layers=3_width=32/'
    return path

def build_histogram_plots(directory,row_num,col_num,prefix,bxy,dz,bxz,theta,phi,q_fac,required_n,estimator):


    path = build_path(bxy,dz,bxz,theta,phi)
    fn =f'{prefix}_'+ build_suffix(q_fac,required_n,estimator)+'.jpg'
    source = 'data_100/'+path+fn
    dest = directory + f'/r={row_num}_c={col_num}.jpg'
    shutil.copy(source,dest)

if __name__ == '__main__':
    est_comb = ['real_weights','NCE_Q','real_TRE_Q','rulsif']
    bxys = [0.0,0.1,0.25,0.5]
    ns = [1000,5000,10000]
    beta_xz = 0.5
    combs = [[1,2.0,2.0],[3,3.0,2.0],[15,8.0,2.0],[50,16.0,2.0]]
    qs  =[0.25,0.5,0.75,1.0]
    prefix='pvalhsit'

    for est in est_comb:
        for bxy in bxys:
            for q in qs:
                directory = f'{est}_by={bxy}_q={q}'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                else:
                    shutil.rmtree(directory)
                    os.makedirs(directory)
                for r,n in enumerate(ns):
                    for c,comb in enumerate(combs):
                        dz = comb[0]
                        theta = comb[1]
                        phi = comb[2]
                        try:
                            build_histogram_plots(directory=directory,
                                                  prefix=prefix,
                                                  bxy=bxy,
                                                  bxz=beta_xz,
                                                  dz=dz,
                                                  theta=theta,
                                                  phi=phi,
                                                  q_fac=q,
                                                  required_n=n,
                                                  estimator=est,
                                                  row_num=r,
                                                  col_num=c)
                            print('success')
                        except Exception as e:
                            print(e)

