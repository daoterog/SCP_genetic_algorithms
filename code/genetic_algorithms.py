from auxiliaries import check_factibility, calculatecosts
from neighborhoods import find_neighborhoods
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
    sorted_keys = individual.sort_values()

    df_copy = df.copy()

    # Add subsets in the order specified by the random keys and 
    # stop when factibility is reached
    for subset in sorted_keys.index.tolist():
        subsets.append(subset)
        
        subset_elements = df_copy[df_copy[subset] == 1].index
        df_copy.drop(subset_elements, inplace = True, axis = 0)
        df_copy.drop([subset], inplace = True, axis = 1)

        if df_copy.empty:
            break

    individual_aux = pd.Series(np.zeros(len(individual)))
    individual_aux[subsets] = 1

    z = calculatecosts(subsets, costs)
    print(z)

    return individual_aux, z

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
    for index, row in population.iterrows():

        population.loc[index,:], z = generate_solution(df, costs, row)
        zs.append(z)

    zs = pd.Series(zs)

    return population, zs

def tournament(zs):

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
    for i in range(2):
        child1 = zs.sample(1).index.tolist()[0]
        child2 = zs.sample(1).index.tolist()[0]

        while child1 == child2 and parents.count(child1) == 0 and parents.count(child2) == 0:
            child2 = zs.sample(1).index.tolist()[0]

        if zs[child1] < zs[child2]:
            parents.append(child1)
        else:
            parents.append(child2)

    return parents

def make_factible(df, costs, subsets_child, neigh = 1, n = 10, n1 = 10, n2 = 10, alpha = 0.3):

    """
    Make a non-factible solution factible.

    Args:
        df: DataFrame with elements and subsets
        costs: Series costs of choosing each subset
        subsets_child: array with subsets of child

    Output:
        child: factible solution.
    """

    subsets = find_neighborhoods(df, costs, subsets_child, neigh, n, n1, n2, alpha)

    return subsets
        
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

    # Child1
    child1_parent1 = population.iloc[parents[0],1:crosspoint]
    child1_parent2 = population.iloc[parents[1],crosspoint:]
    subsets_child1_parent1 = child1_parent1[child1_parent1 == 1].index.tolist()
    subsets_child1_parent2 = child1_parent2[child1_parent2 == 1].index.tolist()
    subsets_child1 = subsets_child1_parent1

    df_copy = df.copy()

    subset_elements = df_copy[df_copy[subsets_child1] == 1].index
    df_copy.drop(subset_elements, axis = 0, inplace = True)
    df_copy.drop(subsets_child1, axis = 1, inplace = True)

    # Append subset of second parent if it is not redundant
    for subset in subsets_child1_parent2:

        nelements = df_copy.sum()

        if not (nelements[subset] == 0):
            subsets_child1.append(subset)

            # Update dataframe
            subset_elements = df_copy[df_copy[subset] == 1].index
            df_copy.drop(subset_elements, axis = 0, inplace = True)
            df_copy.drop(subset, axis = 1, inplace = True)

    # Check factibitility and fix if not factible
    if not check_factibility(df, subsets_child1):
        print('NOT FEASIBLE')
        child1 = make_factible(df, costs, subsets_child1)
        print('FOUND SOLUTION')
    else:
        print('FEASIBLE')
        child1 = subsets_child1
        print('FOUND SOLUTION')

    # Child2
    child2_parent1 = population.iloc[parents[1],1:crosspoint]
    child2_parent2 = population.iloc[parents[0],crosspoint:]
    subsets_child2_parent1 = child2_parent1[child2_parent1 == 1].index.tolist()
    subsets_child2_parent2 = child2_parent2[child2_parent2 == 1].index.tolist()
    subsets_child2 = subsets_child2_parent1

    df_copy = df.copy()

    subset_elements = df_copy[df_copy[subsets_child2] == 1].index
    df_copy.drop(subset_elements, axis = 0, inplace = True)
    df_copy.drop(subsets_child2, axis = 1, inplace = True)

    # Append subset of second parent if it is not redundant
    for subset in subsets_child2_parent2:

        nelements = df_copy.sum()

        if not (nelements[subset] == 0):
            subsets_child2.append(subset)

            # Update dataframe
            subset_elements = df_copy[df_copy[subset] == 1].index
            df_copy.drop(subset_elements, axis = 0, inplace = True)
            df_copy.drop(subset, axis = 1, inplace = True)

    if not check_factibility(df, subsets_child2):
        print('NOT FEASIBLE')
        child2 = make_factible(df, costs, subsets_child2)
        print('FOUND SOLUTION')
    else:
        print('FEASIBLE')
        child2 = subsets_child2
        print('FOUND SOLUTION')

    child1_aux = pd.Series(np.zeros(len(costs)))
    child1_aux[child1] = 1
    z1 = calculatecosts(child1, costs)
    print(z1)

    child2_aux = pd.Series(np.zeros(len(costs)))
    child2_aux[child2] = 1
    z2 = calculatecosts(child2, costs)
    print(z2)

    zs = pd.Series([z1,z2])

    return [child1_aux, child2_aux], zs

def mutation(df, costs, childs):

    child1 = childs[0]; child2 = childs[1]

    child1 = make_factible(df, costs, child1)
    child1_aux = pd.Series(np.zeros(len(costs)))
    child1_aux[child1] = 1
    z1 = calculatecosts(child1, costs)
    print(z1)

    child2 = make_factible(df, costs, child2)
    child2_aux = pd.Series(np.zeros(len(costs)))
    child2_aux[child2] = 1
    z2 = calculatecosts(child2, costs)
    print(z2)

    zs = pd.Series([z1,z2])

    return [child1_aux, child2_aux], zs

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

    population, zs = generate_population(df, costs, npop)
    print('Best initial solution %s' % zs.sort_values().iloc[0])

    start_time = time.perf_counter()
    done = False

    ngen = 0

    while not done:
        ngen += 1

        print('GENERATION N%s' % ngen)
        child_count = 0
        for child in range(1,n_childs,2):
            child_count += 2

            parents = tournament(zs)
            childs, z = crossover(df, costs, population, parents)
            print('%s CHILDS HAVE BORN' % child_count)

            rand = np.random.uniform()

            if rand < pmut:
                print('MUTATION')
                childs, z = mutation(df, costs, childs)
            
            population = population.append(pd.Series(childs[0]), ignore_index = True)
            population = population.append(pd.Series(childs[1]), ignore_index = True)
            zs = zs.append(z, ignore_index = True)

        print('END OF BREEDING\nMAY THE STRONGER SURVIVE')

        best = zs.sort_values().iloc[:npop].index
        population = population.iloc[best,:].reset_index(drop = True)
        zs = zs.iloc[best].reset_index(drop = True)

        time_now = time.perf_counter() - start_time
        if time_now > max_time:
            break

    return population, zs