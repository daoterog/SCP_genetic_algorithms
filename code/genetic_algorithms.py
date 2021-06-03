from .auxiliaries import check_factibility, calculatecosts
from .neighborhoods import find_neighborhoods
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

        # Check for redundancy
        nelements = df_copy.sum()
        if nelements[subset] > 0:
            subsets.append(subset)

            # Update DataFrame
            subset_elements = df_copy[df_copy[subset] == 1].index
            df_copy.drop(subset_elements, inplace = True, axis = 0)
        
        # Update DataFrame even if subset is no considered
        df_copy.drop([subset], inplace = True, axis = 1)

        # Break when there a no elements left to asign 
        if df_copy.empty:
            break

    # Create new individual
    individual_aux = pd.Series(np.zeros(len(individual)))
    individual_aux[subsets] = 1

    # Calculate costs
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
        parents: list with parents.
    """

    # Generate childs
    parents = []
    for i in range(2):

        # Select randomly from sample
        parent1 = zs.sample(1).index.tolist()[0]
        parent2 = zs.sample(1).index.tolist()[0]

        # Check if they are the same
        while parent1 == parent2 or parents.count(parent2) != 0 or parents.count(parent2) != 0:
            parent2 = zs.sample(1).index.tolist()[0]

        # The best parent is the first parent
        if zs[parent1] < zs[parent2]:
            parents.append(parent1)
        else:
            parents.append(parent2)

    return parents

def make_factible(df, costs, subsets_child, neigh = 4, n = 10, n1 = 10, n2 = 10, alpha = 0.3):

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

def crossoverV3(df, costs, population, parents):

    # Extract the parents
    parent1 = population.iloc[parents[0],:]
    parent2 = population.iloc[parents[1],:]

    # Make the mix
    gene_mix = parent1 + parent2
    child_subsets = gene_mix[gene_mix > 0].index.tolist()

    child1 = []; child2 = []
    done = False

    df_copy = df.copy()

    for subset in child_subsets:

        if not done:
            nelements = df_copy.sum()

            if not (nelements[subset] == 0):
                child1.append(subset)

                # Update dataframe
                subset_elements = df_copy[df_copy[subset] == 1].index
                df_copy.drop(subset_elements, axis = 0, inplace = True)
                
            df_copy.drop(subset, axis = 1, inplace = True)

            # Break if solution is already reached
            if df_copy.empty:
                done = True
                df_copy = df.copy()

        else:
            nelements = df_copy.sum()

            if not (nelements[subset] == 0):
                child2.append(subset)

                # Update dataframe
                subset_elements = df_copy[df_copy[subset] == 1].index
                df_copy.drop(subset_elements, axis = 0, inplace = True)
                
            df_copy.drop(subset, axis = 1, inplace = True)

            # Break if solution is already reached
            if df_copy.empty:
                break

    # Improve the solution
    child1 = make_factible(df, costs, child1)
    child1 = make_factible(df, costs, child1)
    child2 = make_factible(df, costs, child2)
    child2 = make_factible(df, costs, child2)

    child1_aux = pd.Series(np.zeros(len(parent1)))
    child1_aux[child1] = 1
    z1 = calculatecosts(child1, costs)

    child2_aux = pd.Series(np.zeros(len(parent2)))
    child2_aux[child2] = 1
    z2 = calculatecosts(child2, costs)

    zs = pd.Series([z1,z2])

    return [child1_aux, child2_aux], zs

def crossoverV2(df, costs, population, parents):

    # Extract the parents
    parent1 = population.iloc[parents[0],:]
    parent2 = population.iloc[parents[1],:]

    # Make the mix
    gene_mix = parent1 + parent2

    # Child1 initial subsets
    child1_subsets = gene_mix[gene_mix == 2].index.tolist()

    # Check feasibility and fix if not factible
    if not check_factibility(df, child1_subsets):
        print('NOT FEASIBLE')
        child1_subsets = make_factible(df, costs, child1_subsets)
        child1_subsets = make_factible(df, costs, child1_subsets)
        child1_subsets = make_factible(df, costs, child1_subsets)
    else:
        print('FEASIBLE')
        print('IMPROVEMENT')
        child1_subsets = make_factible(df, costs, child1_subsets)
        child1_subsets = make_factible(df, costs, child1_subsets)

    child1 = pd.Series(np.zeros(len(parent1)))
    child1[child1_subsets] = 1
    z1 = calculatecosts(child1, costs)

    # Child2 initial subsets
    child2_subsets = gene_mix[gene_mix == 1].index.tolist()

    # Check factibitility and fix if not factible
    if not check_factibility(df, child2_subsets):
        print('NOT FEASIBLE')
        child2_subsets = make_factible(df, costs, child2_subsets)
        child2_subsets = make_factible(df, costs, child2_subsets)
        child2_subsets = make_factible(df, costs, child2_subsets)
    else:
        print('FEASIBLE')
        print('IMPROVEMENT')
        child2_subsets = make_factible(df, costs, child2_subsets)
        child2_subsets = make_factible(df, costs, child2_subsets)
    
    child2 = pd.Series(np.zeros(len(parent1)))
    child2[child2_subsets] = 1
    z2 = calculatecosts(child2, costs)

    zs = pd.Series([z1,z2])

    return [child1, child2], zs

def crossoverV1(df, costs, population, parents):

    # Extract the parents
    parent1 = population.iloc[parents[0],:]
    parent2 = population.iloc[parents[1],:]

    # Make the mix
    gene_mix = parent1 + parent2
    child_subsets = gene_mix[gene_mix > 0].index.tolist()

    child1 = []; child2 = []
    alternate = True

    df_copy1 = df.copy()
    df_copy2 = df.copy()

    for subset in child_subsets:

        if alternate and not df_copy1.empty:
            alternate = False

            nelements = df_copy1.sum()

            if not (nelements[subset] == 0):
                child1.append(subset)

                # Update dataframe
                subset_elements = df_copy1[df_copy1[subset] == 1].index
                df_copy1.drop(subset_elements, axis = 0, inplace = True)
                
            df_copy1.drop(subset, axis = 1, inplace = True)

        elif not alternate and not df_copy2.empty:
            alternate = True

            nelements = df_copy2.sum()

            if not (nelements[subset] == 0):
                child2.append(subset)

                # Update dataframe
                subset_elements = df_copy2[df_copy2[subset] == 1].index
                df_copy2.drop(subset_elements, axis = 0, inplace = True)
                
            df_copy2.drop(subset, axis = 1, inplace = True)

        elif df_copy1.empty and df_copy2.empty:
            break

    # Improve the solution
    child1 = make_factible(df, costs, child1)
    child1 = make_factible(df, costs, child1)
    child2 = make_factible(df, costs, child2)
    child2 = make_factible(df, costs, child2)

    child1_aux = pd.Series(np.zeros(len(parent1)))
    child1_aux[child1] = 1
    z1 = calculatecosts(child1, costs)

    child2_aux = pd.Series(np.zeros(len(parent2)))
    child2_aux[child2] = 1
    z2 = calculatecosts(child2, costs)

    zs = pd.Series([z1,z2])

    return [child1_aux, child2_aux], zs

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

    subset_elements = df_copy[(df_copy[subsets_child1] == 1).sum(axis = 1) >= 1].index
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

        # Break if solution is already reached
        if df_copy.empty:
            break

    # Check factibitility and fix if not factible
    if not check_factibility(df, subsets_child1):
        print('NOT FEASIBLE')
        child1 = make_factible(df, costs, subsets_child1)
        child1 = make_factible(df, costs, child1)
        child1 = make_factible(df, costs, child1)
    else:
        print('FEASIBLE\nIMPROVEMENT')
        child1 = make_factible(df, costs, subsets_child1)
        child1 = make_factible(df, costs, child1)

    # Child2
    child2_parent1 = population.iloc[parents[1],1:crosspoint]
    child2_parent2 = population.iloc[parents[0],crosspoint:]
    subsets_child2_parent1 = child2_parent1[child2_parent1 == 1].index.tolist()
    subsets_child2_parent2 = child2_parent2[child2_parent2 == 1].index.tolist()
    subsets_child2 = subsets_child2_parent1

    df_copy = df.copy()

    subset_elements = df_copy[(df_copy[subsets_child2] == 1).sum(axis = 1) >= 1].index
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

        # Break if solution is already reached
        if df_copy.empty:
            break

    if not check_factibility(df, subsets_child2):
        print('NOT FEASIBLE')
        child2 = make_factible(df, costs, subsets_child2)
        child2 = make_factible(df, costs, child2)
        child2 = make_factible(df, costs, child2)
    else:
        print('FEASIBLE\nIMPROVEMENT')
        child2 = make_factible(df, costs, subsets_child2)
        child2 = make_factible(df, costs, child2)

    child1_aux = pd.Series(np.zeros(len(costs)))
    child1_aux[child1] = 1
    z1 = calculatecosts(child1, costs)
    # print(z1)

    child2_aux = pd.Series(np.zeros(len(costs)))
    child2_aux[child2] = 1
    z2 = calculatecosts(child2, costs)
    # print(z2)

    zs = pd.Series([z1,z2])

    return [child1_aux, child2_aux], zs

def mutation(df, costs, childs):

    """
    Perform an extra mutation in the solution.

    Args:
        df: DataFrame with data
        costs: pandas Series with costs
        childs: list with children

    Output:
        childs: list with mutated children
    """

    # Define children
    child1 = childs[0]; child2 = childs[1]

    # Perform mutation in Child1
    child1 = make_factible(df, costs, child1)
    child1 = make_factible(df, costs, child1)
    child1_aux = pd.Series(np.zeros(len(costs)))
    child1_aux[child1] = 1
    z1 = calculatecosts(child1, costs)
    # print(z1)

    # Perform mutation in Child2
    child2 = make_factible(df, costs, child2)
    child2 = make_factible(df, costs, child2)
    child2_aux = pd.Series(np.zeros(len(costs)))
    child2_aux[child2] = 1
    z2 = calculatecosts(child2, costs)
    # print(z2)

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

    # Start counting time
    start_time = time.perf_counter()
    done = False

    # Generate initial population
    population, zs = generate_population(df, costs, npop)
    print('Best initial solution %s' % zs.sort_values().iloc[0])

    # Generations count
    ngen = 0
    i = 1
    best1 = 0

    while not done:
        ngen += 1
        print('GENERATION N%s' % ngen)

        child_count = 0

        # Start breeding the children
        while child_count < n_childs:
            child_count += 2

            # Choose parents
            parents = tournament(zs)

            # Perform crossover of the parents
            if i == 1:
                childs, z = crossover(df, costs, population, parents)
                print('%s CHILDS HAVE BORN' % child_count)
            elif i == 2:
                childs, z = crossoverV1(df, costs, population, parents)
                print('%s CHILDS HAVE BORN' % child_count)
            elif i == 3:
                childs, z = crossoverV3(df, costs, population, parents)
                print('%s CHILDS HAVE BORN' % child_count)
            else:
                childs, z = crossoverV2(df, costs, population, parents)
                print('%s CHILDS HAVE BORN' % child_count)

            rand = np.random.uniform()

            # Decide whether a mutation is performed or not
            if rand < pmut:
                print('MUTATION')
                childs, z = mutation(df, costs, childs)
            
            # Append the children to the population and solution DataFrame
            population = population.append(pd.Series(childs[0]), ignore_index = True)
            population = population.append(pd.Series(childs[1]), ignore_index = True)
            zs = zs.append(z, ignore_index = True)

        print('END OF BREEDING\nMAY THE STRONGER SURVIVE')

        # Select npop best solutions and eliminate the others
        best = zs.sort_values().iloc[:npop].index

        if best1 == zs.iloc[best.tolist()[0]]:
            print('change i')
            if i < 4:
                i += 1
            else:
                i = 1
            
            # Generate initial population
            print('CONTAMINATE POPULATION')
            cont_population, cont_zs = generate_population(df, costs, n_childs)
            population = population.append(cont_population, ignore_index = True)
            zs = zs.append(cont_zs, ignore_index = True)
        else:
            best1 = zs.iloc[best.tolist()[0]]
            i = 1
            print(best1)

        print(i)
        population = population.iloc[best,:].reset_index(drop = True)
        zs = zs.iloc[best].reset_index(drop = True)

        print('New solutions %s' % zs.sort_values())

        # Check time restriction
        time_now = time.perf_counter() - start_time
        if time_now > max_time:
            break

    z_min = zs.iloc[0]
    best_solution = population.iloc[0,:]
    best_subsets = best_solution[best_solution == 1].index.tolist()

    best_subsets = [subset + 1 for subset in best_subsets]

    return z_min, best_subsets