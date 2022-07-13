# Dual Mirror Descent for Online Allocation Problems
This repo consists Pytorch code for the OR paper "[The best of many worlds: Dual mirror descent for online allocation problems](https://pubsonline.informs.org/doi/abs/10.1287/opre.2021.2242?journalCode=opre)".

## Experiment 1: Online Linear Programming

nrm.py contains useful functions for run the experiments. time_period.py, resource_dimension.py, and decision_dimension.py are the code to run the experiments and generate output in csv files for the three subfigures in Figure 1, respectively. Here are the documents for the optional arguments in these three files:

optional arguments:
  -h, --help            show this help message and exit
  --T		       The number of samples.
  --num_trials	       The number of random trials.
  --num_params	       The number of random parameters.
  --step_size_constant  The step size is step_size_constant/sqrt{T}.
  --reference	       Reference function in mirror descent
  --budget_ratio	       The ratio between budget and the average consumption.


## Experiment 2: Proportional Matching

adx-alloc-data-2014 folder contains the data file generated following the procedure of "Yield Optimization of Display Advertising with Ad Exchange", Management Science. The dataset has 12 advertisers and 100,000 impressions. pub2-ads.txt contains the value of rho for each advertiser. pub2-sample.txt contains the revenue of matching each impression to the corresponding advertiser. We rescale the revenue so that the largest term is 1 in our experiment. We rescale \rho such that sum_j rho_j =1.5.

main.py is the main file to run the experiments, and save the output in csv files. Here are the documents for each optional arguments:

optional arguments:
  -h, --help            show this help message and exit
  --num_trials NUM_TRIALS
                        The number of random trials.
  --lambd LAMBD         The coefficient of regularizer.
  --data_name DATA_NAME
                        The name of the dataset. Must be pub1-pub7.
  --step_size_constant STEP_SIZE_CONSTANT
                        The step-size constant.
  --regularizer REGULARIZER
                        The regularizer r.
  --reference REFERENCE
                        The reference function h.
  --save_frequency SAVE_FREQUENCY
                        How many iterations we save for the output.
  --T_ending T_ENDING   The number of samples.
  --num_T NUM_T         The number of T values.
  --sum_rho SUM_RHO     The sum of rho.


### Dependency:
cvxpy                              1.0.31
pandas                             0.20.2
numpy                              1.16.6