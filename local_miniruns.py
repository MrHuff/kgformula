from kgformula.utils import simulation_object,simulation_object_hsic
from generate_job_params import *
import os
def run_jobs(args):
    if args['job_type']=='kc':
        j = simulation_object(args)
    elif args['job_type']=='hsic':
        j = simulation_object_hsic(args)
    j.run()
    del j


if __name__ == '__main__':
    input = {
        'idx':0,
        'ngpu':1,
        'job_folder': 'exp_jobs_hsic'
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
        job_params['device'] = 0
        job_params['unique_job_idx'] = idx%ngpu
        run_jobs(args=job_params)
