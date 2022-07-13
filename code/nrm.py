import numpy as np
import pandas as pd
import argparse
import sys
from scipy.stats import entropy
import collections
import scipy as sp
from scipy import sparse

np.set_printoptions(threshold=sys.maxsize)
EPS = 1e-9
np.random.seed(123)
np.set_printoptions(precision=3)

FBAR = 10

class Sample(object):
  def __init__(self, consumption, revenue):
    self.consumption = consumption
    self.revenue = revenue

class Dataset(object):
  def __init__(self, m, d, beta_parameters=(1,3), error_variance=0.1, budget_ratio=0.5):
    self.m = m
    self.d = d
    self.prob = (1+np.random.beta(beta_parameters[0], beta_parameters[1], m))/2
    self.error_variance = error_variance
    self.rho = np.random.uniform(low=budget_ratio/2, high=(1+budget_ratio)/2, size=m) * self.prob
    a = np.random.normal(0, 1, m)
    self.theta = np.abs(a) / np.linalg.norm(a)

  def create_sample(self):
    consumption = sparse.csr_matrix(np.array([np.random.binomial(1, prob, self.d) for prob in self.prob]))
    revenue = consumption.transpose().dot(self.theta) + np.random.normal(0, self.error_variance, self.d)
    revenue = np.clip(revenue, 0, FBAR)
    return Sample(consumption, revenue)

class Algorithm(object):
  def __init__(self, reference, step_size_constant, T, data):
    
    self.reference = reference
    self.T = T
    self.data = data
    self.d = data.d
    self.m = data.m
    if self.reference in ["square_norm", "scaled_square_norm"]:
      self.eta = step_size_constant / np.sqrt(T)
    elif self.reference in ["entropy"]:
      self.eta = step_size_constant / np.sqrt(T)
    elif self.reference in ["projected_entropy"]:
      self.eta = step_size_constant / np.sqrt(T)

  # Computes argmax{x\inX} {f_t(x)-bmu^T x}
  def compute_x(self, sample, dual):
    reduced_cost = sample.revenue - sample.consumption.transpose().dot(dual)
    index = np.argmax(reduced_cost)
    x = np.zeros(self.d)
    if reduced_cost[index] > 0:
      x[index] = 1

    return x, index

  # Computes argmax{x\inX} {f_t(x)-bmu^T x}
  def compute_x_with_budget(self, sample, dual, remaining_budget):
    reduced_cost = sample.revenue - sample.consumption.transpose().dot(dual)
    sorted_index = (np.argsort(reduced_cost))[::-1]

    x = np.zeros(self.d)

    for j in range(self.d):
      index = sorted_index[j]
      if reduced_cost[index] <= 0:
        break
      if np.all(sample.consumption[:, index] <= remaining_budget):
        x[index] = 1
        return x, index

    return x, index

  def compute_next_dual(self, dual, gradient):
    # Compute the next iteration: argmin_{dual' in D} eta*gradient*dual' + V_h(dual', dual).
    if self.reference=="square_norm":
      return np.maximum(dual - self.eta * gradient, np.zeros(len(dual)))  
    elif self.reference=="scaled_square_norm":
      return np.maximum(dual - self.eta * gradient/self.data.rho/self.data.rho, np.zeros(len(dual)))
    elif self.reference=="entropy":
      return dual * np.exp(-self.eta*gradient)
    elif self.reference=="projected_entropy":
      rho = self.data.rho
      temp = rho * dual * np.exp(-self.eta*gradient)
      if np.sum(temp) <= FBAR:
        answer = temp / rho
      else:
        answer = FBAR * temp / np.sum(temp) / rho
      return answer

  def mirror_descent(self):
    T = self.T
    m = self.m
    remaining_budget = T * self.data.rho
    if self.reference in ["square_norm", "scaled_square_norm"]:
      ini_dual = np.zeros(m)
    elif self.reference in ["entropy"]:
      ini_dual = 1 / np.sqrt(m) * np.ones(m)
    elif self.reference in ["projected_entropy"]:
      ini_dual = 1 / np.sqrt(m) * np.ones(m) * self.data.rho

    ind_revenue = np.zeros(m)
    ind_consumption = np.zeros(m)
    cum_revenue = np.array([0])
    cum_reg_revenue = np.array([0])
    current_dual = np.copy(ini_dual)
    sum_dual = np.copy(ini_dual)
    all_remain_budget = np.empty((0, m))

    for t in range(T):
      sample = self.data.create_sample()
      x, index = self.compute_x(sample, current_dual)
      
      if np.all(remaining_budget >= sample.consumption.dot(x)) and x[index] != 0:
        remaining_budget -= sample.consumption.dot(x)
        cum_revenue = np.append(cum_revenue, cum_revenue[t] + sample.revenue[index])
      else:
        cum_revenue = np.append(cum_revenue, cum_revenue[t])

      gradient = - sample.consumption * x + self.data.rho
      current_dual = self.compute_next_dual(current_dual, gradient)
      sum_dual += current_dual
    return cum_revenue[-1], sum_dual/(T), all_remain_budget

  def compute_optimal_obj(self, optimal_dual):
    T = self.T
    m = self.m
    dual_obj = T * self.data.rho.dot(optimal_dual)
    for t in range(T):
      sample = self.data.create_sample()
      dual_obj += np.max([np.max(sample.revenue - sample.consumption.transpose().dot(optimal_dual)), 0])
      # print(dual_obj)
    return dual_obj

def load_data(instance_name, input_model, ergodic_level):
  """
  Load the data.
  """
  
  if input_model == "iid":
    directory = "../adx-alloc-data-2014/"
    revenue_file_name = directory + instance_name + "-sample.txt"
  elif input_model == "ergodic":
    directory = "../adx-alloc-data-2014-ergodic/"
    revenue_file_name = directory + instance_name + "-corr"+ergodic_level+"-sample.txt"
  revenue = np.loadtxt(revenue_file_name, delimiter=",")
  revenue = revenue / np.max(revenue)

  rho_file_name = "../adx-alloc-data-2014/" + instance_name + "-ads.txt"
  rho = np.array([])
  with open(rho_file_name) as f:
    lines = f.readlines()
  for line in lines:
    rho = np.append(rho, float(line.split(":")[2][:-4]))
  return Dataset(revenue, rho)

def random_trials(algorithm, num_trials):
  revenue_coll = np.array([])
  opt_dual_coll = np.array([])

  for i in range(num_trials):
    np.random.seed(123+i)
    final_reward, ave_dual, remaining_budget = algorithm.mirror_descent()
    np.random.seed(123+i)
    optimal_dual_obj = algorithm.compute_optimal_obj(ave_dual)
    revenue_coll = np.append(revenue_coll, final_reward)
    opt_dual_coll = np.append(opt_dual_coll, optimal_dual_obj)

  return revenue_coll, opt_dual_coll, opt_dual_coll - revenue_coll

def random_params(m, d, T, beta_parameters, error_variance, budget_ratio, step_size_constant, reference, num_trials, num_params):
  revenue_colls = np.array([])
  opt_dual_colls = np.array([])
  regret_colls = np.array([])

  for i in range(num_params):
    np.random.seed(100+i)
    data = Dataset(m=m, d=d, beta_parameters=beta_parameters, error_variance=error_variance, budget_ratio=budget_ratio)
    algorithm = Algorithm(reference, step_size_constant=step_size_constant, T=T, data=data)
    revenue_coll, opt_dual_coll, regret_coll = random_trials(algorithm, num_trials)
    revenue_colls = np.append(revenue_colls, revenue_coll)
    opt_dual_colls = np.append(opt_dual_colls, opt_dual_coll)
    regret_colls = np.append(regret_colls, regret_coll)

  return revenue_colls, opt_dual_colls, regret_colls



if __name__ == "__main__":
  """
  Run multiple random trials of Algorithm 3 in the main paper for solving the
  proportional matching problems with high entropy, and save the regret to the
  output csv file.
  """
  
  data = Dataset(m=100, d=10, beta_parameters=(3,1), error_variance=2, budget_ratio=0.5)
  # print(data.rho)
  algorithm = Algorithm("projected_entropy", step_size_constant=10, T=1000, data=data)
  for i in range(1):
    np.random.seed(123+i)
    final_reward, ave_dual, remaining_budget = algorithm.mirror_descent()
    np.random.seed(123+i)
    optimal_dual_obj = algorithm.compute_optimal_obj(ave_dual)

  revenue_coll, opt_dual_coll, regret_coll = random_trials(algorithm, 2)
  print(regret_coll)
