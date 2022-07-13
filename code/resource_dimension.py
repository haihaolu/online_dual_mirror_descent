import numpy as np
import pandas as pd
import argparse

import nrm

parser = argparse.ArgumentParser()
parser.add_argument("--T", help="The number of samples.", type=int, default=1000)
parser.add_argument("--d", help="The dimension of decision variable.", type=int, default=10)
parser.add_argument("--num_trials", help="The number of random trials.", type=int, default=2)
parser.add_argument("--num_params", help="The number of random parameters.", type=int, default=2)
parser.add_argument("--step_size_constant", help="The step-size constant.", type=float, default=1)
parser.add_argument("--reference", help="The reference function h.", type=str, default="square_norm")
parser.add_argument("--m_ending", help="The largest number of m.", type=int, default=1000)
parser.add_argument("--num_m", help="The number of m values.", type=int, default=2)
parser.add_argument("--budget_ratio", help="The ratio between budget and the average consumption.", type=int, default=0.5)
args = parser.parse_args()

target_ms = np.linspace(101, args.m_ending, num=args.num_m, dtype=int)
rows = []
columns = ["m", "Revenue", "Regret", "std"]

for target_m in target_ms:

  revenue_coll, opt_dual_coll, regret_coll = nrm.random_params(m=target_m, d=args.d, T=args.T, beta_parameters=(3,1), error_variance=1, budget_ratio=args.budget_ratio, step_size_constant=args.step_size_constant, reference=args.reference, num_trials=args.num_trials, num_params=args.num_params)
  ave_revenue = np.average(revenue_coll)
  ave_regret = np.average(regret_coll)
  std = np.std(regret_coll)

  rows.append([target_m, ave_revenue, ave_regret, std])

df = pd.DataFrame(rows, columns=columns)
df.to_csv("/results/over_m_"+args.reference+"_step_size_constant_"+str(args.step_size_constant).replace('.', "")+"_num_trials_"+str(args.num_trials)+"_num_params_"+str(args.num_params)+"_m_ending"+str(args.m_ending)+".csv", index=False)
