#backdoor-HSIC (bd_HSIC) 

Accompanying code for the paper "A Kernel Test for Causal Association via Noise Contrastive Backdoor Adjustment". 

To generate the synthetic data used in the paper, the general procedure is the following:

1. Generate the necessary data 
2. Generate and run the appropriate job using the "interpreter_run.py"
3. Post-process the results
4. Generate plots from the results

We list which scripts to run and which job to run to recreate the experiments in the paper.

| Experiment               | File to generate data         | File to run experiments        | Post process file           | Generate plot               |
|--------------------------|-------------------------------|--------------------------------|-----------------------------|-----------------------------|
| Binary                   | generate_simple_null.py       | do_null_binary                 | kchsic_categorical_power.py | kchsic_categorical_power.py |
| Linear continuous        | generate_data_multivariate.py | kc_rule_new                    | post_process.py             | create_plots.py             |
| Mixed                    | generate_mixed_data.py        | do_null_mix                    | post_process.py             | create_plots.py             |
| Non-linear continuous    | generate_hdm_breaker.py       | hdm_breaker_fam_y=i_100, i=1,4 | post_process.py             | create_plots.py             |
| Linear kernel binary     | generate_simple_null.py       | do_null_binary_linear_kernel   | kchsic_categorical_power.py | kchsic_categorical_power.py |
| Linear kernel continuous | generate_data_multivariate.py | linear                         | post_process.py             | create_plots.py             |

To get the results for the benchmark post-double-selection method and the R scripts can be run. To run the R scripts, some .pt files need to be converted to .csv. To do this, use the pt_to_csv.py script.
For the real world experiments, do the following:

1. Lalonde - Run lalonde_real_experiment.py
2. Twins - Process the data using twins_preprocess.py, then run twins_real_experiment.py

This repository will include the results needed to recreate the plots in the paper already.
