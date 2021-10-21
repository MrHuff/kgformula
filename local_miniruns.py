from kgformula.utils import *
from generate_job_params import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="5"

def run_jobs(args):
    if args['job_type']=='kc':
        j = simulation_object(args)
    elif args['job_type']=='hsic':
        j = simulation_object_hsic(args)
    elif args['job_type']=='kc_rule_new':
        j = simulation_object_rule_new(args)
    elif args['job_type']=='regression':
        j = simulation_object_linear_regression(args)
    j.run()
    del j


if __name__ == '__main__':
    input = {
        'idx':0,#bugbug
        'ngpu':48,
        'job_folder': 'kc_rule_1d_linear_0.5_3'
    }
    listjob = os.listdir(input['job_folder'])
    # for i in range(len(listjob)):
    # input['idx']=i
    idx = input['idx']
    ngpu = input['ngpu']
    fold = input['job_folder']
    jobs = os.listdir(fold)
    jobs.sort()
    for el in jobs:
        job_params = load_obj(el,folder=f'{fold}/')
        job_params['device'] = 0
        job_params['unique_job_idx'] = 99999
        # if (job_params['estimator']=='rulsif') and (job_params['est_params']['separate']==False):
        # print(job_params)
        try:
            run_jobs(args=job_params)
        except Exception as e:
            print(e)