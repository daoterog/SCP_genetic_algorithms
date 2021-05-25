import pandas as pd 
import numpy as np

def lowerbound(df, costs):

    """
    Calculate lower bound for the problem. Calculate the minimum cost of coverage 
    for every element and sum it.

    Args:
        df: Dataframe that specifies which subset cover which elements 
        costs: Series costs of choosing each subset

    Output:
        lb: lower bound
    """
    lb = 0
    nelements = df.sum()

    for i in range(df.shape[0]):
        elem_i = df.loc[i,:]
        subsets = elem_i[elem_i == 1].index
        subsets_cost = costs[subsets]
        subsest_nelem = nelements[subsets]
        ratio = subsets_cost/subsest_nelem
        min_ratio = ratio.min()
        lb += min_ratio

    return int(np.ceil(lb))

def calculatecosts(subsets, costs):

    """
    Calculate solution cost function.

    Args:
        subsets: chosen subsets
        costs: subsets cost

    Output:
        z: cost
    """

    z = costs[subsets].sum()

    return z


def check_factibility(df, subsets):

    """
    Check factibility of a solution.

    Args:
        df: DataFrame with elements and subsets
        subsets: solution

    Output:
        factible: Boolean response
    """

    df_copy = df.copy()

    subset_elements = df_copy[(df_copy[subsets] == 1).sum(axis = 1) >= 1].index
    df_copy.drop(subset_elements, axis = 0, inplace = True)

    factible = df_copy.empty

    return factible
    
def data(c_nsub, g_nsub, n_nsub, c_subset, g_subset, n_subset):
    c_max = max(c_nsub)
    g_max = max(g_nsub)
    n_max = max(n_nsub)
    
    def_max = max([c_max, g_max, n_max])
    len_files = len(c_nsub)

    matrix_c = np.zeros((len_files, def_max))
    matrix_g = np.zeros((len_files, def_max))
    matrix_n = np.zeros((len_files, def_max))

    i = 0
    while i < len_files:
    
        # Extract subset constructive
        sub = c_subset[i]

        # Rewrite matrix
        for j in range(len(sub)):
            matrix_c[i][j] = sub[j]
            top = j
        for k in range(top, def_max):
            matrix_c[i][k] = None

        # Extract subset grasp
        sub = g_subset[i]

        # Rewrite matrix
        for j in range(len(sub)):
            matrix_g[i][j] = sub[j]
            top = j
        for k in range(top, def_max):
            matrix_g[i][k] = None

        # Extract subset noise
        sub = n_subset[i]

        # Rewrite matrix
        for j in range(len(sub)):
            matrix_n[i][j] = sub[j]
            top = j
        for k in range(top, def_max):
            matrix_n[i][k] = None
    
        i +=1

    return matrix_c, matrix_g, matrix_n