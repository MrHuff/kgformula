from kgformula.utils import *
from generate_job_params import *
import os

def run_jobs(args):
    if args['job_type']=='kc':
        j = simulation_object(args)
    elif args['job_type']=='kc_px':
        j = simulation_object_px(args)
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
        'idx':0,
        'ngpu':48,
        'job_folder': 'random_uniform'
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
        #Index related error, might just add a try clause so it doesn't break
        print(job_params)
        run_jobs(args=job_params)
