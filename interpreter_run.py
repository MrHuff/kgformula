from kgformula.utils import simulation_object
import argparse
from generate_job_params import *
parser = argparse.ArgumentParser()
parser.add_argument('--idx', type=int, default=0, help='cdim')
parser.add_argument('--ngpu', type=int, default=4, help='cdim')
parser.add_argument('--job_folder', type=str, default='', help='cdim')

def run_jobs(args):
    j = simulation_object(args)
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
