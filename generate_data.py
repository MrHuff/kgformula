from kgformula.utils import generate_data

if __name__ == '__main__':
    seeds = 100
    for n in [10000]:
        for y_a in [0.0]:
            for y_b in [0.0]:
                for z_a in [0.0]:
                    for z_b in [0.0,1e-3,1e-2,0.05,0.1,0.25,0.5,1]:
                        for cor in [0.5]:
                            for q_fac in [0.6,0.7,0.8]:
                                generate_data(y_a=y_a,y_b=y_b,z_a=z_a,z_b=z_b,cor=cor,n=n,seeds=seeds,theta=4,q_factor=q_fac)
