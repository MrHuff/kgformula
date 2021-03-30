from kgformula.utils import simulation_object,simulation_object_hsic,simulation_object_GCM,simulation_object_linear_regression,simulation_object_adaptive
from generate_job_params import *
import os

def run_jobs(args):
    if args['job_type']=='kc':
        j = simulation_object(args)
    elif args['job_type']=='kc_adaptive':
        j = simulation_object_adaptive(args)
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
        'job_folder': 'kc_hsic_break_test_adaptive_5'
    }
    listjob = os.listdir(input['job_folder'])
    for i in range(len(listjob)):
        input['idx']=i
        idx = input['idx']
        ngpu = input['ngpu']
        fold = input['job_folder']
        jobs = os.listdir(fold)
        jobs.sort()
        print(jobs[idx])
        job_params = load_obj(jobs[idx],folder=f'{fold}/')
        job_params['device'] = 1
        job_params['unique_job_idx'] = idx%ngpu
        job_params['n'] = 10000

        print(job_params)
        run_jobs(args=job_params)
