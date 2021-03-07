from kgformula.utils import simulation_object,simulation_object_hsic,simulation_object_GCM
import argparse
from generate_job_params import *
parser = argparse.ArgumentParser()
parser.add_argument('--idx', type=int, default=0, help='cdim')
parser.add_argument('--ngpu', type=int, default=4, help='cdim')
parser.add_argument('--job_folder', type=str, default='', help='cdim')

def run_jobs(args):
    if args['job_type']=='kc':
        j = simulation_object(args)
    elif args['job_type']=='hsic':
        j = simulation_object_hsic(args)
    elif args['job_type']=='gcm':
        j = simulation_object_GCM(args)
    j.run()
    del j

if __name__ == '__main__':
    input  = vars(parser.parse_args())
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
