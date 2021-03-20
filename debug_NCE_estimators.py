from kgformula.utils import simulation_object,simulation_object_hsic,simulation_object_GCM,simulation_object_linear_regression
from generate_job_params import *
import os
#TODO understand why a well trained estimator gives wrong weights and undertrained estimator gives right power...
#TODO WEIRDEST BUG EVER HOW THE FUCK DOES CHANGING THE LEARNING RATE AFFECT INITIALIZATION?!?!?!?! No its completely random
# P-value calculation, can we assume its symmetric?! Or this assumption incorrect?! Do a tail adjustment/classifier, i.e. rule based on tail direction.
#We observed this twice already!

#Is it overfitting?
def run_jobs(args):
    if args['job_type']=='kc':
        j = simulation_object(args)
    elif args['job_type']=='hsic':
        j = simulation_object_hsic(args)
    elif args['job_type']=='gcm':
        j = simulation_object_GCM(args)
    elif args['job_type']=='regression':
        j = simulation_object_linear_regression(args)
    j.run()
    del j

if __name__ == '__main__':
    input = {
        'idx':5,
        'ngpu':1,
        'job_folder': 'exp_jobs_kc_est'
    }
    listjob = os.listdir(input['job_folder'])
    idx = input['idx']
    ngpu = input['ngpu']
    fold = input['job_folder']
    jobs = os.listdir(fold)
    jobs.sort()
    print(jobs[idx])
    job_params = load_obj(jobs[idx],folder=f'{fold}/')
    job_params['device'] = 0
    job_params['unique_job_idx'] = idx%ngpu
    print(job_params)
    job_params['est_params']['lr'] = 1e-3
    job_params['est_params']['max_its'] = 10
    job_params['est_params']['kappa'] = 100
    run_jobs(args=job_params)
