
from kgformula.post_process_plots import *
from kgformula.utils import x_q_class
import os
from generate_job_params import *
import ast
from generate_data_multivariate import generate_sensible_variables,calc_snr
from pylab import *
rc('text', usetex=True)

if __name__ == '__main__':
    test = np.random.rand(1000)
    title = "test"
    estimator='real weights'
    xlabl = r'Estimator: {est}$\quad \beta_{XY}={bxy}$'.format(est=str(estimator), bxy=1.0, XY='{XY}')
    plt.hist(test,25)
    plt.suptitle(title)
    plt.xlabel(xlabl)
    plt.savefig('test.png')
