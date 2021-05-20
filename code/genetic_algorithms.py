import numpy as np
import pandas as pd
import time

def generate_population(df, npop):
    """
    Generates random keys in order to create indiviuals.

    Args: 
        df: DataFrame with the elements and subsets.
        npop: number of individuals to create.

    Output:
        population: matrix of random keys, each row represents an inidivual
    """

    m = df.shape[1]

    population = pd.DataFrame(np.random.rand(npop, m))

    for i in range(npop):
        for j in range(m):

            

    return population

def GA(df, npop, max_time):

    initial_population = generate_population

    start_time = time.perf_counter()
    done = False

    niter = 0

    while not done:
        niter += 1



