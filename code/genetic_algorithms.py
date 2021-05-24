from auxiliaries import check_factibility, calculatecost, check_redundancy
import numpy as np
import pandas as pd
import time

def generate_solution(df, costs, individual):

    """
    Generates factibles solution using sorted random keys.

    Args:
        df: DataFrame with elements and subsets
        inidividual: random keys

    Output:
        subsets = factible solution
    """

    subsets = []
    individual_copy = individual.copy()
    sorted_keys = individual.sort_values()

    # Add subsets in the order specified by the random keys and 
    # stop when factibility is reached
    for key in sorted_keys.index.tolist():
        subsets.append(key)
        individual[key] = 1
        factible = check_factibility(df, subsets)

        if factible:
            break

    individual[individual != 1] = 0

    z = caluclatecost(subsets, costs)

    return individual, z

def generate_population(df, costs, npop):
    """
    Generates random keys in order to create indiviuals.

    Args: 
        df: DataFrame with the elements and subsets.
        npop: number of individuals to create.

    Output:
        population: solution matrix, each row represents an inidivual
        zs: list with cost function of each solution
    """

    m = df.shape[1]

    # Generate random keys matrix
    population = pd.DataFrame(np.random.rand(npop, m))

    # Cost function of each solution
    zs = []

    # Convert random keys into binary array
    for i in range(npop):
        individual = population.iloc[i,:]

        population.iloc[i,:], z = generate_solution(df, costs, individual)
        zs.append(z)

    return population, zs

def tournament(npop, zs):

    """
    Chooses parents to perform crossover.

    Args:
        npop: number of individuals in the population
        zs: list with costs of each individual

    Output:
        parents: parents.
    """

    # Generate childs
    parents = []
    for i in range(1):
        child1 = np.random.randint(npop)
        child2 = np.random.randint(npop)

        while child1 == child2 and parents.count(child1) == 0 and parents.count(child2) == 0:
            child2 = np.random.randint(npop)

        if zs[child1] < zs[child2]:
            parents.append(child1)
        else:
            parents.append(child2)

    return parents

def make_factible(df, costs, subsets_child):

    """
    Make a non-factible solution factible.

    Args:
        df: DataFrame with elements and subsets
        costs: Series costs of choosing each subset
        subsets_child: array with subsets of child

    Output:
        child: factible solution.
    """

    subsets = subsets_child

    df_copy = df.copy()
    costs_copy = costs.copy()

    while not check_factibility(df, subsets):

        # Create aux arrays
        nelements = df_copy.sum()

        # Delete substes that no longer have elements
        no_elements = nelements[nelements == 0].index.tolist()
        if no_elements != []:
            df_copy.drop(no_elements, axis = 1, inplace = True)
            costs_copy.drop(no_elements, inplace = True)
            nelements = df_copy.sum()

        # Select subset with the most elements
        max_elements = nelements.max()
        bigger_subsets = nelements[nelements == max_elements].index.tolist()
        



def crossover(df, costs, population, parents):

    """
    Perform crossover with the two parents.

    Output:
        df: DataFrame with elements and subsets
        costs: Series costs of choosing each subset
        population: dataframe with population
        parents: list with index of parents

    Args:
        childs: children with the population.
    """

    # Find crosspoint
    crosspoint = np.random.randint(df.shape[1]-2)+1

    # Make the mix
    child1_parent1 = population.iloc[parents[0],1:crosspoint]
    child1_parent2 = population.iloc[parents[1],crosspoint:]
    subsets_child1_parent1 = child1_parent1[child1_parent1 == 1].index.tolist()
    subsets_child1_parent2 = child1_parent2[child1_parent2 == 1].index.tolist()
    subsets_child1 = subsets_child1_parent1

    for subset in subsets_child1_parent2:

        if not check_redundancy(df, subsets_child1, subset):
            subsets_child1.append(subset)

    child2 = (population.iloc[parents[1],1:crosspoint].tolist() + 
        population.iloc[parents[0],crosspoint:].tolist())
    
    # Check factibitility and fix if not factible
    if not check_factibility(child1):
        child1 = make_factible(df, costs, child1)

    if not check_factibility(child2):
        child2 = make_factible(df, costs, child2)


def GA(df, costs, npop, max_time, n_childs, pmut):

    """
    Genetic algorithm

    Agrs:
        df: DataFrame with data
        costs: pandas Series with costs
        npop: number of individuals in the population
        max_time: maximum time of running
        n_cilds: number of childs
        pmut: probability of mutation

    Output:
        subsets: solution
    """

    initial_population, zs = generate_population(df, costs, npop)

    start_time = time.perf_counter()
    done = False

    ngen = 0

    while not done:
        ngen += 1

        for child in range(n_childs):
            parents = tournament(npop, zs)
            childs = crossover(df, initial_population, parents)

            



