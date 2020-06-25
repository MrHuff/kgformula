import torch
from scipy.stats import kstest
import matplotlib.pyplot as plt
if __name__ == '__main__':
    for n in [100,1000,10000]:
            # f'multivariate_1000/beta_xy=[0, 0]_d_X=3_d_Y=3_d_Z=3_n={n}_yz=0.5_beta_XZ=0.5_theta=8_phi=2.83/'
        seed_max=100
        nr_of_gpus = 5
        estimate = True
        new = False
        estimator = 'classifier'
        val_rate = max(1e-2, 10. / n)
        width = 2048
        layers = 3
        job_name = f'layers={layers}_width={width}/'
        theta = 3
        phi = round(theta**0.5,2)
        beta_xz = 0.5
        d_Z = 3
        # data_path = f'univariate_100_seeds/ground_truth=H_0_y_a=0.0_y_b=0.0_z_a=0.0_z_b=0.5_cor=0.5_n={n}_seeds={seed_max}_{theta}_{phi}/'
        data_path = f'multivariate_100/beta_xy=[0, 0]_d_X=3_d_Y=3_d_Z={d_Z}_n={n}_yz=0.5_beta_XZ={beta_xz}_theta={theta}_phi={phi}/'

        est_params = {'lr': 1e-4,
                               'max_its': 10000,
                               'width': width,
                               'layers': layers,
                               'mixed': False,
                               'bs_ratio': 10. / n,
                               'kappa': 10,
                               'val_rate': val_rate,
                               'n_sample': 250,
                               'criteria_limit': 0.05,
                               'kill_counter': 10,
                               'reg_lambda': 1e-2,
                               'alpha': 0.5}

        residual = seed_max % nr_of_gpus
        interval_size = (seed_max - residual) / nr_of_gpus
        files_to_concat = []
        for i in range(nr_of_gpus):
            if i == nr_of_gpus - 1:
                seeds_a = int(i * interval_size)
                seeds_b = int((i + 1) * interval_size + residual)
                suffix = f'_new={new}_seeds={seeds_a}_{seeds_b}_estimate={estimate}_estimator={estimator}'

            else:
                seeds_a = int(i * interval_size)
                seeds_b = int((i + 1) * interval_size)
                suffix = f'_new={new}_seeds={seeds_a}_{seeds_b}_estimate={estimate}_estimator={estimator}'
            if estimate:
                if estimator == 'kmm':
                    lamb = est_params['reg_lambda']
                    suffix = suffix + f'lambda={lamb}'
                elif estimator == 'classifier':
                    hsic_pval_list = []
                    for key, val in est_params.items():
                        suffix = suffix + f'_{key}={val}'
            files_to_concat.append('p_val_array'+suffix+'.pt')
        concat_tensor = []

        for f in files_to_concat:
            load_data = data_path+job_name+f
            dat = torch.load(load_data)
            concat_tensor.append(dat)
        test_this = torch.cat(concat_tensor,dim=0).numpy()
        stat,pval =kstest(test_this,'uniform')
        print('stat: ',stat, 'pval: ',pval)
        text_file = open(data_path+job_name+"ks_test.txt", "w")
        text_file.write(f'stat: {stat} pval: {pval}')
        text_file.close()
        plt.hist(test_this,bins=25)
        plt.savefig(data_path+job_name+'histogram.png')
        plt.clf()


