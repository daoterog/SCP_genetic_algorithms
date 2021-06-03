import pandas as pd
import numpy as np
import time
import os

from readfile import readfile
from genetic_algorithms import GA
from algorithms import VND, LS

from auxiliaries import lowerbound, data

"""
This file run the selected algorithms over every dataset and compiles the results
into multiple csv files. In order to run correctly the program the user must be 
working inside the /code directory, otherwise, the program won't run. 
"""

# File to read
datasets_path = os.path.join(os.path.dirname(os.getcwd()), 'datasets')
files = ['scp41.txt', 'scp42.txt', 'scpnrg1.txt', 'scpnrg2.txt', 'scpnrg3.txt', 'scpnrg4.txt', 
         'scpnrg5.txt', 'scpnrh1.txt', 'scpnrh2.txt', 'scpnrh3.txt', 'scpnrh4.txt', 'scpnrh5.txt']

# Parameters
npop = 10 # Suggested 
nchilds = 6
pmut = 0.7
maxtime = 300 # In seconds 

# Results to dataframe
ga_scores = []
ga_nsub = []
ga_subset = []
ga_rel = []
ga_times = []
vnd_scores = []
vnd_nsub = []
vnd_subset = []
vnd_rel = []
vnd_times = []
ls_scores = []
ls_nsub = []
ls_subset = []
ls_rel = []
ls_times = []
lbound = []
nelements = []
nsubsets = []
means = ['Mean']

# The first two values printed on the list correspond to the total cost and
# the numbers of subsets chosen respectively
for name in files:

    # Read file
    path = os.path.join(datasets_path, name)
    df, costs = readfile(path)
    nelements.append(df.shape[0])
    nsubsets.append(df.shape[1])

    # Lower bound
    lb = lowerbound(df, costs)
    lbound.append(lb)

    # VND
    print('GA')
    start_ga = time.perf_counter()
    ga_cost, ga_subsets = GA(df, costs, npop, maxtime, nchilds, pmut)
    time_ga = time.perf_counter() - start_ga
    print('C:\t',[ga_cost,len(ga_subsets)] + ga_subsets,'\t',time_ga)
    ga_scores.append(ga_cost)
    ga_rel.append(np.float32(np.round(ga_cost/lb,3)))
    ga_times.append(np.float32(np.round(time_ga,5)))
    ga_nsub.append(len(ga_subsets))
    ga_subset.append(ga_subsets)

    # SA
    print('VND')
    start_vnd = time.perf_counter()
    vnd_cost, vnd_subsets = VND(df, costs)
    time_vnd = time.perf_counter() - start_vnd
    print('C:\t',[vnd_cost,len(vnd_subsets)] + vnd_subsets,'\t',time_vnd)
    vnd_scores.append(vnd_cost)
    vnd_rel.append(np.float32(np.round(vnd_cost/lb,3)))
    vnd_times.append(np.float32(np.round(time_vnd,5)))
    vnd_nsub.append(len(vnd_subsets))
    vnd_subset.append(vnd_subsets)

    # LS
    print('LS')
    start_ls = time.perf_counter()
    ls_cost, ls_subsets = LS(df, costs, neigh = 4, nsol = 30)
    time_ls = time.perf_counter() - start_ls
    print('C:\t',[ls_cost,len(ls_subsets)] + ls_subsets,'\t',time_ls)
    ls_scores.append(ls_cost)
    ls_rel.append(np.float32(np.round(ls_cost/lb,3)))
    ls_times.append(np.float32(np.round(time_ls,5)))
    ls_nsub.append(len(ls_subsets))
    ls_subset.append(ls_subsets)

# Create dataframe
front_ga = np.transpose([ga_scores, ga_nsub])
front_vnd = np.transpose([vnd_scores, vnd_nsub])
front_ls = np.transpose([ls_scores, ls_nsub])
matrix_ga, matrix_vnd, matrix_ls = data(ga_nsub, vnd_nsub, ls_nsub, ga_subset, vnd_subset, ls_subset)
data_front_ga = np.c_[front_ga,matrix_ga]
data_front_vnd = np.c_[front_vnd,matrix_vnd]
data_front_ls = np.c_[front_ls,matrix_ls]

df_ga = pd.DataFrame(data = data_front_ga)
df_vnd = pd.DataFrame(data = data_front_vnd)
df_ls = pd.DataFrame(data = data_front_ls)

means = means + [np.mean(nelements), np.mean(nsubsets), np.mean(lb), np.mean(ga_scores),
              np.mean(ga_rel), np.mean(ga_times), np.mean(vnd_scores), np.mean(vnd_rel),
              np.mean(vnd_times), np.mean(ls_scores), np.mean(ls_rel), np.mean(ls_times)]

columns = ['Files', 'Elements', 'Subsets', 'LB', 'Scores_GA', 'Gap_GA', 'Time_GA', 
            'Scores_VND', 'Gap_VND', 'Time_VND', 'Scores_LS', 'Gap_LS', 'Time_LS']

last = dict(zip(columns, means))
dat = [files, nelements, nsubsets, lbound, ga_scores, ga_rel, ga_times, vnd_scores, vnd_rel, 
        vnd_times, ls_scores, ls_rel, ls_times]
results = pd.DataFrame(data = np.transpose(dat), columns = columns)
results = results.append(last, ignore_index = True)

results_path = os.path.join(os.path.dirname(os.getcwd()), 'results')

# To csv
df_ga.to_csv(os.path.join(results_path,'SCP_DanielOtero_GA.csv'), header = False, index = False)
df_vnd.to_csv(os.path.join(results_path,'SCP_DanielOtero_VND.csv'), header = False, index = False)
df_ls.to_csv(os.path.join(results_path,'SCP_DanielOtero_LS.csv'), header = False, index = False)
results.to_csv(os.path.join(results_path,'SCP_DanielOtero_comparison.csv'), index = False)
