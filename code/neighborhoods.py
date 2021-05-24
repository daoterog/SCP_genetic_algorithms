import pandas as pd 
import numpy as np 
import time

from auxiliaries import calculatecosts

def find_lower_cost_subsets(df, costs, reference_cost):
    df_copy1 = df.copy()
    costs_copy1 = costs.copy()

    # Define child evaluation statistics
    childs_costs = 0
    childs_elements = 0

    # Subset array
    subsets = []

    while childs_costs < reference_cost or not df_copy1.empty:

        # Create aux arrays
        nelements = df_copy1.sum()

        # Delete substes that no longer have elements
        no_elements = nelements[nelements == 0].index.tolist()
        if no_elements != []:
            df_copy1.drop(no_elements, axis = 1, inplace = True)
            costs_copy1.drop(no_elements, inplace = True)
            nelements = df_copy1.sum()
        
        # Breaking if there are no more elements to asign
        if df_copy1.empty:
            break

        # Extract random subset with minimum cost
        min_cost = costs_copy1.min()
        min_cost_subsets = costs_copy1[costs_copy1 == min_cost]
        indx_random_min_cost_subset = min_cost_subsets.sample(1).index[0]

        # Adding subset to list
        subsets.append(indx_random_min_cost_subset)

        # Extract number of elements that covers the lowest costing subset
        nelem_min_cost_subset = nelements[indx_random_min_cost_subset]

        # Sum cost and number of elements of the lowest costing subset
        childs_costs += min_cost
        childs_elements += nelem_min_cost_subset

        # Update df and costs by deleteing chosen subset
        # Extract all the elements contained in the subset and drop them according
        # to their index
        subset_elements = df_copy1[df_copy1[indx_random_min_cost_subset] == 1].index
        df_copy1.drop(subset_elements, axis = 0, inplace = True)
        df_copy1.drop(indx_random_min_cost_subset, axis = 1, inplace = True)
        costs_copy1.drop(indx_random_min_cost_subset, inplace = True)

    return subsets

def solution_generator(df, costs):
    df_copy = df.copy()
    costs_copy = costs.copy()

    # Subset array
    subsets = []

    while not df_copy.empty:

        # Create aux arrays
        nelements = df_copy.sum()

        # Delete substes that no longer have elements
        no_elements = nelements[nelements == 0].index.tolist()

        if no_elements != []:

            # Drop empty subsets
            df_copy.drop(no_elements, axis = 1, inplace = True)
            costs_copy.drop(no_elements, inplace = True)

            # Update nelements
            nelements = df_copy.sum()

        # Find maximum number of elements that a subset has
        maximum_element = nelements.max()

        # Find subsets that have the most elements and its costs
        nmax_subsets = nelements[nelements == maximum_element].index
        cost_nmax_subset = costs_copy[nmax_subsets]

        # Find subsets with minimum cost
        mincost = cost_nmax_subset.min()
        mincost_nmax_subsets = cost_nmax_subset[cost_nmax_subset == mincost]

        # Pick random
        mincost_subset_idx = mincost_nmax_subsets.sample(1).index[0]

        # Assign solution 
        subsets.append(mincost_subset_idx)

        # Update Dataframe
        # Extract all the elements contained in the subset and drop them according
        # to their index
        subset_elements = df_copy[df_copy[mincost_subset_idx] == 1].index
        df_copy.drop(subset_elements, axis = 0, inplace = True)
        df_copy.drop(mincost_subset_idx, axis = 1, inplace = True)
        costs_copy.drop(mincost_subset_idx, inplace = True)

    return subsets

def first_neighborhood(df, costs, subsets):

    """
    First way of iterating over a neighborhood. The method identifies the subset with
    with the maximum cost, if there are multiple it chooses the one that covers the most
    elements, if there is a tie then it chooses one randomly. Then the function 
    find_subset_set attemps to find a set of substes that cover the elements left 
    uncovered by the possible elimination of the chosen subset. 

    Args:
        df: Dataframe that specifies which subset cover which elements 
        costs: costs of choosing each subset
        subsests: chosen subsets

    Output:
        subsets: new chosen subsets
    """

    # Extract maximum cost subset
    subset_costs = costs.iloc[subsets]
    max_cost = subset_costs.max()
    subsets_max_cost = subset_costs[subset_costs == max_cost].index.tolist()

    # In the case that there are various subsets with the same cost
    # pick the one with the most elements
    if len(subsets_max_cost) != 1:
        nelements = df.sum()
        nelem_subsets_max_cost = nelements.iloc[subsets_max_cost]
        subset_max_elements = nelem_subsets_max_cost.max()
        max_elements_subset = nelem_subsets_max_cost[nelem_subsets_max_cost == subset_max_elements]

        # Pick one randomly
        subsets_max_cost = max_elements_subset.sample(1).index[0]

    else:
        subsets_max_cost = subsets_max_cost[0]

    return subsets_max_cost, max_cost

def second_neighborhood(df, costs, n, subsets):

    """
    Second way of iterating through a neighborhood. The method identifies the subsets
    with the n largest costs and attemps to found a set of subsets that covers the elements
    left out if those subsets are not considered with the help of the find_subset_set function.

    Args:
        df: DataFrame that specifies which subset cover which elements 
        costs: costs of choosing each subset
        n: specifies the number of subsets considered
        subsests: chosen subsets

    Output:
        subsets: new chosen subsets
    """

    # Extract the n maximum cost subset
    subset_costs = costs.iloc[subsets]
    max_cost = subset_costs.nlargest(n)
    subsets_cost = max_cost.sum()
    subsets_max_cost = max_cost.index.tolist()

    return subsets_max_cost, subsets_cost

def F_S_neighborhood(df, costs, subsets, neigh, n = 2):

    """
    Interface that connects with the way of extracting first and second neighborhood

    Args:
        df: Dataframe that specifies which subset cover which elements 
        costs: costs of choosing each subset
        subsests: chosen subsets
        neigh: number that indicates which neighborhood to head
        n: n condition for second neighborhood

    Output:
        subsets: new chosen subsets
    """

    # Copy df and costs to perform operations
    df_copy = df.copy()
    costs_copy = costs.copy()

    # Aux
    subsets_max_cost = []
    max_cost = 0
    cutted_subsets = [] 

    # Decide which neighborhood structure to use
    if neigh == 1:
        subsets_max_cost, max_cost = first_neighborhood(df, costs, subsets)

        # Instance new subset array
        cutted_subsets = [s for s in subsets if s!= subsets_max_cost]

    else:
        subsets_max_cost, max_cost = second_neighborhood(df, costs, n, subsets)

        # Instance new subset array
        cutted_subsets = [s for s in subsets if  not s in subsets_max_cost]

    # Update Dataframe
    # Extract all the elements contained in the subset and drop them according
    # to their index
    subset_elements = df_copy[(df_copy[cutted_subsets] == 1).sum(axis = 1) >= 1].index
    df_copy.drop(subset_elements, axis = 0, inplace = True)
    df_copy.drop(cutted_subsets, axis = 1, inplace = True)
    costs_copy.drop(cutted_subsets, inplace = True)

    # If the DataFrame is empty then we could easily improve the solution,
    # if not, we improve it using the following funtcion
    if not df_copy.empty:
        replacement_subsets = find_lower_cost_subsets(df_copy, costs_copy, max_cost)

        if replacement_subsets:
            # print('REPLACEMENT FOUND')

            subsets = cutted_subsets + replacement_subsets
        # else:
        #     print('REPLACEMENT NOT FOUNT')
    else:
        # print('NO REPLACEMENT NEEDED')

        subsets = cutted_subsets

    return subsets

def third_neighborhood(df, costs, n, subsets):

    """
    Third way of iterating through the neighborhoods. This method takes the n subsets
    with the largest costs and attemps to find a set of of subsets that covers the elements
    left out when each of this subsets is not considered individually. The function then chooses
    to replace the subsets that lowers the most the cost function, if there is a tie it chooses randomly.

    Args:
        df: DataFrame that specifies which subset cover which elements 
        costs: costs of choosing each subset
        n: specifies the number of subsets considered
        subsests: chosen subsets

    Output:
        subsets: new chosen subsets
    """

    # Copy df and costs to perform operations
    df_copy = df.copy()
    costs_copy = costs.copy()

    # Extract maximum cost subset
    subset_costs = costs_copy.iloc[subsets]
    max_cost = subset_costs.nlargest(n)

    # To store results
    zs = []
    subset_options = []

    # Start loop
    for subsets_max_cost in max_cost.index.tolist():
            
        # Copy df and costs in order to iterate over "previous" versions
        # not current tests
        df_copy1 = df_copy.copy()
        costs_copy1 = costs_copy.copy()

        # Extract cost of the subset taken into account
        cost = max_cost[subsets_max_cost]

        zs, subset_options = T_F_add_option(df_copy1, costs_copy1, subsets, subsets_max_cost, 
                                            cost, zs, subset_options, 3)

    # Datatype conversions in order to make operations easier
    zs = pd.Series(zs)
    min_zs = zs.min()
    mins = zs[zs == min_zs]
    rand_min = mins.sample(1).index[0]

    subsets = subset_options[rand_min]

    return subsets

def fourth_neighborhood(df, costs, n, alpha, subsets):

    """
    Fourth way of iterating over a neighborhood. It first selects the top half subsets with higher
    costs. Then it runs n iterations in which a random sample of size alpha*len(tophalf) will be considered.
    The find_subset_set will be find a set of subsets that cover the elements left when not considering
    the random sample. Finally, after n iterations, it selects the one that minimizes the cost function, if
    there is a tie it selects one randomly.

    Args:
        df: DataFrame that specifies which subset cover which elements. 
        costs: costs of choosing each subset.
        n: specifies the number of subsets considered.
        alpha: percentage of the top half subsets that will be considered.
        subsests: chosen subsets.

    Output:
        subsets: new chosen subsets.
    """

    # Copy df and costs to perform operations
    df_copy = df.copy()
    costs_copy = costs.copy()

    # Extract the toph half maximum cost subset
    subset_costs = costs_copy.iloc[subsets]
    sorted_subset_costs = subset_costs.sort_values()
    half = round(len(sorted_subset_costs)/2)
    top_half_subset_costs = sorted_subset_costs.iloc[half:]

    # To store results
    zs = []
    subset_options = []

    for i in range(n):

        # Copy df and costs in order to iterate over "previous" versions
        # not current tests
        df_copy1 = df_copy.copy()
        costs_copy1 = costs_copy.copy()

        # Generate random sample of top half maximum cost subset
        n_sample = round(len(top_half_subset_costs)*alpha)
        max_cost = top_half_subset_costs.sample(n_sample)
        subsets_cost = max_cost.sum()
        subsets_max_cost = max_cost.index.tolist()

        zs, subset_options = T_F_add_option(df_copy1, costs_copy1, subsets, subsets_max_cost, 
                                            subsets_cost, zs, subset_options, 4)

    # Datatype conversions in order to make operations easier
    zs = pd.Series(zs)
    min_zs = zs.min()
    mins = zs[zs == min_zs]
    rand_min = mins.sample(1).index[0]

    subsets = subset_options[rand_min]

    return subsets

def T_F_add_option(df, costs, subsets, subsets_max_cost, max_cost, zs, subset_options, neigh):

    """
    Helper function to add options to array for third and fourth neighborhood.

    Args:
        df: DataFrame that specifies which subset cover which elements 
        costs: costs of choosing each subset
        subsests: chosen subsets
        subsets_max_cost: subsets array with maximum cost. Each array is chosen differently by nighborhood
        max_cost: subsets_max_cost cost
        zs: array use to append cost function values
        subsets_options: array used to append subsets options

    Output:
        zs and subsets_options with option added
    """

    df_copy = df.copy()
    costs_copy = costs.copy()

    # Aux
    cutted_subsets = []
    
    # Instance new subset array
    if neigh == 3:
        cutted_subsets = [s for s in subsets if s!= subsets_max_cost]
    else:
        cutted_subsets = [s for s in subsets if  not s in subsets_max_cost]

    # Update Dataframe
    # Extract all the elements contained in the subset and drop them according
    # to their index
    subset_elements = df_copy[(df_copy[cutted_subsets] == 1).sum(axis = 1) >= 1].index
    df_copy.drop(subset_elements, axis = 0, inplace = True)
    df_copy.drop(cutted_subsets, axis = 1, inplace = True)
    costs_copy.drop(cutted_subsets, inplace = True)

    # If the DataFrame is empty then we could easily improve the solution,
    # if not, we improve it using the following funtcion
    if not df_copy.empty:
        replacement_subsets = find_lower_cost_subsets(df_copy, costs_copy, max_cost)

        if replacement_subsets:
            # print('REPLACEMENT FOUND')

            # Calculate subset option and store it
            subsets_option = cutted_subsets + replacement_subsets
            new_option = calculatecosts(subsets_option, costs)

            zs.append(new_option)
            subset_options.append(subsets_option)
        # else:
        #     print('REPLACEMENT NOT FOUNT')
    else:
        # print('NO REPLACEMENT NEEDED')

        # Calculate subset option and store it
        subsets_option = cutted_subsets
        new_option = calculatecosts(subsets_option, costs)

        zs.append(new_option)
        subset_options.append(subsets_option)

    return zs, subset_options

def find_neighborhoods(df, costs, subsets, neigh, n, n1, n2, alpha):

    """
    Function used to redirect the search to its corresponding neighborhood

    Args:
        df: Dataframe that specifies which subset cover which elements 
        costs: costs of choosing each subset
        subsests: chosen subsets
        neigh: number that indicates which neighborhood to head
        n: n condition for second neighborhood
        n1: n condition for third neighborhood
        n2: n condition for fourth neighborhood
        alpha: percentage of the top half subsets that will be considered.

    Output:
        subsets: newly chosen subsets
    """

    # Select Neigborhood
    if neigh == 1:
        subsets = F_S_neighborhood(df, costs, subsets, neigh = 1)

    elif neigh == 2:
        subsets = F_S_neighborhood(df, costs, subsets, neigh = 2, n = n)

    elif neigh == 3:
        subsets = third_neighborhood(df, costs, n1, subsets)

    else:
        subsets = fourth_neighborhood(df, costs, n2, alpha, subsets)

    return subsets

def LS_neighborhoods(df, costs, subsets, neigh, n, n1, n2, alpha):

    """
    Method that initializes and performs the descent in the local search. It stops when it reaches
    a local optimum.

    Args:
        df: Dataframe that specifies which subset cover which elements 
        costs: costs of choosing each subset
        subsests: chosen subsets
        neigh: number that indicates which neighborhood to head
        n: n condition for second neighborhood
        n1: n condition for third neighborhood
        n2: n condition for fourth neighborhood
        alpha: percentage of the top half subsets that will be considered.

    Output:
        subsets: newly chosen subsets
    """

    before = calculatecosts(subsets, costs)
    new = 0
    
    # Initialize Search
    while before > new:
        before = calculatecosts(subsets, costs)

        subsets = find_neighborhoods(df, costs, subsets, neigh, n, n1, n2, alpha)
        
        new =  calculatecosts(subsets, costs)
        print("New Solution: %s" % new)
    
    return subsets

def aux_neighborhoods(df, costs, subsets, neigh, n, n1, n2, alpha, nsol):

    """
    Helper function that is incharged of running the next iteration over the neighborhood
    nsol times. Due to the fact that randomness is involved in each way of generating new solutions
    this action is necessary.

    Args:
        df: Dataframe that specifies which subset cover which elements 
        costs: costs of choosing each subset
        subsests: chosen subsets
        neigh: number that indicates which neighborhood to head
        n: n condition for second neighborhood
        n1: n condition for third neighborhood
        n2: n condition for fourth neighborhood
        alpha: percentage of the top half subsets that will be considered.
        nsol: number of times that it will run

    Output:
        subsets: newly chosen subsets
    """

    # Start time
    start_time = time.perf_counter()

    # To store results
    zs = []
    subset_options = []

    for i in range(nsol):

        # Find solution that belongs to the j neighborhood
        new_subsets = find_neighborhoods(df, costs, subsets, neigh, n, n1, n2, alpha)
        new_cost = calculatecosts(new_subsets, costs)

        zs.append(new_cost)
        subset_options.append(new_subsets)

        # Time counter
        time_now = time.perf_counter() - start_time
        if time_now > 30:
            print('BREAK')
            done = True
            break

    # Datatype conversions in order to make operations easier
    zs = pd.Series(zs)
    min_zs = zs.min()
    mins = zs[zs == min_zs]
    rand_min = mins.sample(1).index[0]

    subsets = subset_options[rand_min]

    return subsets
