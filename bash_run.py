from kgformula.utils import job_parser,run_job_func

if __name__ == '__main__':
    args = vars(job_parser().parse_args())
    run_job_func(args)

