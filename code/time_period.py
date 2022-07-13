import numpy as np
import pandas as pd
import argparse

import nrm

parser = argparse.ArgumentParser()
parser.add_argument("--m", help="The dimension of resources.", type=int, default=100)
parser.add_argument("--d", help="The dimension of decision variable.", type=int, default=10)
parser.add_argument("--num_trials", help="The number of random trials.", type=int, default=2)
parser.add_argument("--num_params", help="The number of random parameters.", type=int, default=2)
parser.add_argument("--step_size_constant", help="The step-size constant.", type=float, default=1)
parser.add_argument("--reference", help="The reference function h.", type=str, default="square_norm")
parser.add_argument("--T_ending", help="The number of samples.", type=int, default=1000)
parser.add_argument("--num_T", help="The number of T values.", type=int, default=2)
parser.add_argument("--budget_ratio", help="The ratio between budget and the average consumption.", type=int, default=0.5)
args = parser.parse_args()

target_Ts = np.linspace(101, args.T_ending, num=args.num_T, dtype=int)
rows = []
columns = ["T", "Revenue", "Regret", "std"]

for target_T in target_Ts:
  revenue_coll, opt_dual_coll, regret_coll = nrm.random_params(m=args.m, d=args.d, T=target_T, beta_parameters=(3,1), error_variance=1, budget_ratio=args.budget_ratio, step_size_constant=args.step_size_constant, reference=args.reference, num_trials=args.num_trials, num_params=args.num_params)

  ave_revenue = np.average(revenue_coll)
  ave_regret = np.average(regret_coll)
  std = np.std(regret_coll)
  print(regret_coll)

  rows.append([target_T, ave_revenue, ave_regret, std])

df = pd.DataFrame(rows, columns=columns)
df.to_csv("results/over_time_"+args.reference+"_step_size_constant_"+str(args.step_size_constant).replace('.', "")+"_num_trials_"+str(args.num_trials)+"_num_params_"+str(args.num_params)+"_T_ending"+str(args.T_ending)+".csv", index=False)

