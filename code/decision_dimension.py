import numpy as np
import pandas as pd
import argparse

import nrm

parser = argparse.ArgumentParser()
parser.add_argument("--T", help="The number of samples.", type=int, default=1000)
parser.add_argument("--m", help="The dimension of resource constraints.", type=int, default=100)
parser.add_argument("--num_trials", help="The number of random trials.", type=int, default=2)
parser.add_argument("--num_params", help="The number of random parameters.", type=int, default=2)
parser.add_argument("--step_size_constant", help="The step-size constant.", type=float, default=1)
parser.add_argument("--reference", help="The reference function h.", type=str, default="square_norm")
parser.add_argument("--d_ending", help="The largest number of m.", type=int, default=1000)
parser.add_argument("--num_d", help="The number of d values.", type=int, default=2)
parser.add_argument("--budget_ratio", help="The ratio between budget and the average consumption.", type=int, default=0.5)
args = parser.parse_args()

target_ds = np.linspace(101, args.d_ending, num=args.num_d, dtype=int)
rows = []
columns = ["d", "Revenue", "Regret", "std"]

for target_d in target_ds:
  revenue_coll, opt_dual_coll, regret_coll = nrm.random_params(m=args.m, d=target_d, T=args.T, beta_parameters=(3,1), error_variance=1, budget_ratio=args.budget_ratio, step_size_constant=args.step_size_constant, reference=args.reference, num_trials=args.num_trials, num_params=args.num_params)
  ave_revenue = np.average(revenue_coll)
  ave_regret = np.average(regret_coll)
  std = np.std(regret_coll)
  print(regret_coll)

  rows.append([target_d, ave_revenue, ave_regret, std])

df = pd.DataFrame(rows, columns=columns)
df.to_csv("results/over_d_"+args.reference+"_step_size_constant_"+str(args.step_size_constant).replace('.', "")+"_num_trials_"+str(args.num_trials)+"_num_params_"+str(args.num_params)+"_d_ending"+str(args.d_ending)+".csv", index=False)

