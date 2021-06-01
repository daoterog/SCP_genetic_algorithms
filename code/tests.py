from readfile import readfile
from genetic_algorithms import GA

import os
import time

datasets_path = os.path.join(os.path.dirname(os.getcwd()), 'datasets')

dataset = 'scp42.txt'

path = os.path.join(datasets_path, dataset)

df, costs = readfile(path)

start_time = time.perf_counter()
population, zs = GA(df, costs, 50, 300, 50, 0.3)
total_time = time.perf_counter() - start_time

print(zs.sort_values())
print(total_time)
