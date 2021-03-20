import os
import GPUtil
try:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Get the first available GPU
    DEVICE_ID_LIST = GPUtil.getAvailable(order='memory', limit=1)
    DEVICE_ID = DEVICE_ID_LIST[0] # grab first element from list
    # Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
except:
    print('no gpu rip')
from kgformula.utils import simulation_object,simulation_object_hsic,simulation_object_GCM,simulation_object_linear_regression
from generate_job_params import *


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
        'idx':1,
        'ngpu':1,
        'job_folder': 'exp_jobs_kc_est_test'
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
    # job_params['est_params']['lr'] = 1e-3
    # job_params['est_params']['max_its'] = 10
    # job_params['est_params']['kappa'] = 10
    # job_params['bootstrap_runs'] = 500
    run_jobs(args=job_params)
