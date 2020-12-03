from kgformula.utils import simulation_object
import GPUtil
import torch.multiprocessing as mp
import math
import os
import argparse
from generate_job_params import *
parser = argparse.ArgumentParser()
parser.add_argument('--idx', type=int, default=0, help='cdim')
parser.add_argument('--ngpu', type=int, default=4, help='cdim')

def run_jobs(args):
    j = simulation_object(args)
    j.run()
    del j

if __name__ == '__main__':
    jobs = os.listdir("job_dir")
    jobs.sort()
    input  = vars(parser.parse_args())
    idx = input['idx']
    ngpu = input['ngpu']
    print(jobs[idx])
    job_params = load_obj(jobs[idx],folder='job_dir/')
    job_params['device'] = 0
    job_params['unique_job_idx'] = idx%ngpu

    run_jobs(args=job_params)
