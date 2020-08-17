from kgformula.utils import generate_data

if __name__ == '__main__':
    seeds = 100
    for n in [1000,10000]:
        for y_a in [0.0]:
            for y_b in [0.0,0.5]:
                for z_a in [0.0]:
                    for z_b in [0.5,0]:
                        for cor in [0.5]:
                            for q_fac in [1e-2,1e-3]:
                                generate_data(y_a=y_a,y_b=y_b,z_a=z_a,z_b=z_b,cor=cor,n=n,seeds=seeds,theta=4,q_factor=q_fac)
