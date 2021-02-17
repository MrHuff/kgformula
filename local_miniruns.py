from kgformula.utils import simulation_object
from generate_job_params import *

def run_jobs(args):
    j = simulation_object(args)
    j.run()
    del j

if __name__ == '__main__':
    input = {
        'idx':0,
        'ngpu':1,
        'job_folder': 'exp_jobs_true_weights_sanity'
    }
    for i in range(8):
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
