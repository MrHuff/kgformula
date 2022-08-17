from kgformula.utils import *
import argparse
from generate_job_params import *
from kgformula.kernel_baseline_utils import *
parser = argparse.ArgumentParser()
parser.add_argument('--idx', type=int, default=0, help='cdim')
parser.add_argument('--ngpu', type=int, default=4, help='cdim')
parser.add_argument('--job_folder', type=str, default='', help='cdim')

def run_jobs(args):
    if args['job_type']=='kc':
        j = simulation_object(args)
    elif args['job_type']=='kc_rule_new':
        j = simulation_object_rule_new(args)
    elif args['job_type']=='kc_rule_correct_perm':
        j = simulation_object_rule_perm(args)
    elif args['job_type']=='hsic':
        j = simulation_object_hsic(args)
    elif args['job_type']=='regression':
        j = simulation_object_linear_regression(args)
    elif args['job_type'] in ['cfme','old_statistic']:
        j = kernel_baseline(args)
    #TODO: Permutation seems somewhat well behaved could be that the weights are just wrong so go through estimation procedure!
    #TODO: Sign-flip wrong
    #TODO: switched 0 and 1 in density ratio interpretation? 1 on top and 0 on bottom (You have it switched!)
    # elif args['job_type']=='old_causal_test':
    #     j = simulation_object_linear_regression(args)
    j.run()
    del j

if __name__ == '__main__':
    input  = vars(parser.parse_args())
    idx = input['idx']
    ngpu = input['ngpu']
    fold = input['job_folder']
    jobs = os.listdir(fold)
    job_params = load_obj(f'job_{idx}.pkl',folder=f'{fold}/')
    print(job_params)
    job_params['device'] = 0
    job_params['unique_job_idx'] = idx
    run_jobs(args=job_params)
