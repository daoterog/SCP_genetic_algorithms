import pandas as pd
import numpy as np
import time

from .neighborhoods import solution_generator, aux_neighborhoods, LS_neighborhoods
from .auxiliaries import calculatecosts

def VND(df, costs, n = 2, n1 = 10, n2 = 10, alpha = 0.3, nsol = 20):

    """
    VND algorithm.

    Args:
        df: Dataframe that specifies which subset cover which elements 
        costs: costs of choosing each subset
        neigh: number that indicates which neighborhood to head
        n: n condition for second neighborhood
        n1: n condition for third neighborhood
        n2: n condition for fourth neighborhood
        alpha: percentage of the top half subsets that will be considered.
        nsol: number of times that it will run

    Output:
        subsets: newly chosen subsets
        cost: cost function
    """

    # Start time
    start_time = time.perf_counter()

    # Generate First Solution and calculate cost
    initial_subsets = solution_generator(df, costs)
    initial_cost = calculatecosts(initial_subsets, costs)

    print('Initial Solution: %s' % initial_cost)

    # Aux
    neigh = 1
    cost_before = initial_cost
    subsets_before = initial_subsets

    # Start neighborhood search
    while neigh <= 4:
        print(neigh)

        # Find solution that belongs to the j neighborhood
        new_subsets = aux_neighborhoods(df, costs, subsets_before, neigh, n, n1, n2, alpha, nsol)
        new_cost = calculatecosts(new_subsets, costs)
        print('New Solution: %s' % new_cost)

        if new_cost < cost_before:
            neigh = 1

            # Update values
            cost_before = new_cost
            subsets_before = new_subsets

            print('NEW IMPROVEMENT')
        
        else:
            neigh += 1

        # Time counter
        time_now = time.perf_counter() - start_time
        if time_now > 300:
            print('BREAK')
            done = True
            break

    print('Final Solution: %s' % cost_before)

    subsets_before = [subset + 1 for subset in subsets_before]
    
    return cost_before, subsets_before


def VNS(df, costs, n = 2, n1 = 10, n2 = 10, alpha = 0.3, nsol = 20):

    """
    VNS algorithm.

    Args:
        df: Dataframe that specifies which subset cover which elements 
        costs: costs of choosing each subset
        neigh: number that indicates which neighborhood to head
        n: n condition for second neighborhood
        n1: n condition for third neighborhood
        n2: n condition for fourth neighborhood
        alpha: percentage of the top half subsets that will be considered.
        nsol: number of times that it will run

    Output:
        subsets: newly chosen subsets
        cost: cost function
    """

    # Generate First Solution and calculate cost
    initial_subsets = solution_generator(df, costs)
    initial_cost = calculatecosts(initial_subsets, costs)

    print('Initial Solution: %s' % initial_cost)

    # Aux
    neigh = 1
    cost_before = initial_cost
    subsets_before = initial_subsets

    # Start neighborhood search
    while neigh <= 4:
        print(j)

        # Find solution that belongs to the j neighborhood
        new_subsets = aux_neighborhoods(df, costs, subsets_before, neigh, n, n1, n2, alpha, nsol)
        new_cost = calculatecosts(new_subsets, costs)
        print('New Solution: %s' % new_cost)

        # More Auxiliaries
        new_cost_before = new_cost
        new_subsets_before = new_subsets
        local_optimum = False

        # Check if it is a local optimum
        while not local_optimum:

            newsub = aux_neighborhoods(df, costs, new_subsets_before, neigh, n, n1, n2, alpha, nsol)
            newc = calculatecosts(newsub, costs)
            print('New Solution: %s' % newc)

            if newc < new_cost_before:

                new_cost_before = newc
                new_subsets_before = newsub
                print('NEW IMPROVEMENT')

            else:

                local_optimum = True
                print('LOCAL OPTIMUM')

        if new_cost_before < cost_before:
            cost_before = new_cost_before
            subsets_before = new_subsets_before
            neigh = 1

        else:
            neigh += 1

    print('Final Solution: %s' % cost_before)

    subsets_before = [subset + 1 for subset in subsets_before]

    return cost_before, subsets_before

def SA(df, costs, T0, Tf, L, r, neigh = 3, n = 2, n1 = 10, n2 = 10, alpha = 0.3, nsol = 20):

    """
    Simulated Anealing algorithm.

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
        cost: cost function
    """

    # Start time
    start_time = time.perf_counter()

    # Generate First Solution and calculate cost
    initial_subsets = solution_generator(df, costs)
    initial_cost = calculatecosts(initial_subsets, costs)

    print('Initial Solution: %s' % initial_cost)

    # Initialize T
    T = T0

    # Aux
    cost_before = initial_cost
    subsets_before = initial_subsets
    best_cost = 300000000000
    best_subsets = []
    done = False

    # Start Loop
    while T > Tf:
        l = 0 

        # Start second Loop
        while l < L:
            l += 1
            new_cost = 0
            new_subsets = []

            # Find solution that belongs to the j neighborhood
            new_subsets = aux_neighborhoods(df, costs, subsets_before, neigh, n, n1, n2, alpha, nsol)
            new_cost = calculatecosts(new_subsets, costs)
            print('New Solution: %s' % new_cost)

            d = new_cost - cost_before

            if d < 0:

                # Update solution
                cost_before = new_cost
                subsets_before = new_subsets
                print('NEW IMPROVEMENT')

                # Store best results
                if best_cost > cost_before:
                    best_cost = cost_before
                    best_subsets = subsets_before

                # Time counter
                time_now = time.perf_counter() - start_time
                if time_now > 300:
                    print('BREAK')
                    done = True
                    break
            
            else:
                rand = np.random.uniform(0,1)

                if rand < np.exp(-d/T):

                    # Update solution
                    cost_before = new_cost
                    subsets_before = new_subsets
                    print('SET BACK')

                    # Time counter
                    time_now = time.perf_counter() - start_time
                    if time_now > 300:
                        print('BREAK')
                        done = True
                        break
        
        T = r*T
        if done: 
            break 

    print('Final Solution: %s' % best_cost)

    best_subsets = [subset + 1 for subset in best_subsets]

    return best_cost, best_subsets

def LS(df, costs, neigh, n = 2, n1 = 10, n2 = 10, alpha = 0.3, nsol = 20):

    """
    Initialize nsol local search and chooses the one with the best results.

    Args:
        df: Dataframe that specifies which subset cover which elements 
        costs: costs of choosing each subset
        neigh: number that indicates which neighborhood to head
        n: n condition for second neighborhood
        n1: n condition for third neighborhood
        n2: n condition for fourth neighborhood
        alpha: percentage of the top half subsets that will be considered
        nsol: number of iterations to run

    Output:
        subsets: newly chosen subsets
    """

    # Start time
    start_time = time.perf_counter()

    # Generate First Solution and calculate cost
    initial_subsets = solution_generator(df, costs)
    initial_cost = calculatecosts(initial_subsets, costs)

    print('Initial Solution: %s' % initial_cost)

    # To store results
    zs = []
    subset_options = []

    for i in range(nsol):

        print('Iteration Number %s' % i)

        subsets_option = LS_neighborhoods(df, costs, initial_subsets, neigh, n, n1, n2, alpha)
        cost_option = calculatecosts(subsets_option, costs)

        zs.append(cost_option)
        subset_options.append(subsets_option)

        # Time counter
        time_now = time.perf_counter() - start_time
        if time_now > 300:
            print('BREAK')
            done = True
            break


    
    # Select minimum, if multiple, pick randomly
    zs = pd.Series(zs)
    min_zs = zs.min()
    mins = zs[zs == min_zs]
    rand_min = mins.sample(1).index[0]

    subsets = subset_options[rand_min]

    print('Final Solution: %s' % min_zs)

    subsets = [subset + 1 for subset in subsets]

    return min_zs, subsets