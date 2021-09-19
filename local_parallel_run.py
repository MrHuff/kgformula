from kgformula.utils import *
from generate_job_params import *
import os
import time
import GPUtil
from random import randint
from torch.multiprocessing import Pool
# torch.multiprocessing.set_start_method("spawn")


def run_a_job(job_name,JOB_FOLDER,device_idx,idx):
    job_params = load_obj(job_name,folder=f'{JOB_FOLDER}/')
    time.sleep(randint(3,7))
    job_params['device'] = device_idx
    job_params['unique_job_idx'] = idx
    # job_params['job_dir'] = JOB_FOLDER
    if job_params['job_type']=='kc':
        j = simulation_object(job_params)
    elif job_params['job_type']=='hsic':
        j = simulation_object_hsic(job_params)
    elif job_params['job_type']=='regression':
        j = simulation_object_linear_regression(job_params)
    elif job_params['job_type'] == 'kc_rule_new':
        j = simulation_object_rule_new(job_params)
    j.run()
    del j
    return(device_idx)

def run_jobs(args):
    if args['job_type']=='kc':
        j = simulation_object(args)
    elif args['job_type']=='kc_rule':
        j = simulation_object_rule(args)
    elif args['job_type']=='hsic':
        j = simulation_object_hsic(args)
    elif args['job_type']=='kc_rule_new':
        j = simulation_object_rule_new(args)
    elif args['job_type']=='regression':
        j = simulation_object_linear_regression(args)
    j.run()
    del j



if __name__ == '__main__':
    print(os.getcwd())
    JOB_FOLDER = 'do_null_mixed_est_3d_2'
    jobs = os.listdir(JOB_FOLDER)
    jobs.sort()
    n_parallel=16
    n_gpu = 8
    inputs = [(el,JOB_FOLDER,i%8,i%(n_parallel*n_gpu)) for i,el in enumerate(jobs)]
#     print(inputs)
    with Pool(processes = n_parallel) as p:   # Paralleizing over 2 GPUs
        results = p.starmap(run_a_job,inputs)