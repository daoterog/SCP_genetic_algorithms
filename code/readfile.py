import pandas as pd
import numpy as np
import os

def readcost(n,lines):
    costs = []
    line = []
    index = 0
    while len(costs) != n:
        line = lines.pop(0).split()
        for i, cost in enumerate(line):
            if len(costs) != n: 
                costs.append(int(cost))
                index = i
            else:
                break
    return (costs, lines, line, index)

def completetable(df, lines):
    element = 0

    l = 0
    c = 0
    while element < df.shape[0] :
        nsubset = int(lines[l].split()[c])
        cont = 0
        c += 1
        while cont < nsubset:
            while c < len(lines[l].split()):
                    if cont < nsubset:
                        subset = int(lines[l].split()[c])
                        df.iloc[element-1,subset - 1] = 1
                        c += 1
                        cont += 1
                    else:
                        break

            if ((cont < nsubset) or (c < len(lines[l].split()))):
                l += 1
                c = 0
        l += 1
        c = 0
        element += 1

    return (df, lines, l, c)

def readfile(path):

    # Read file lines
    file1 = open(path, 'r')
    lines = file1.readlines()

    # Extract dimensions
    first = lines.pop(0).split()
    m = int(first[0])
    n = int(first[1])

    # Create dataframe of zeros
    matrix = np.zeros((m,n))
    df = pd.DataFrame(matrix)

    # Extract costs
    (costs, lines, line, index) = readcost(n,lines)
    
    (df, lines, l, c) = completetable(df, lines)

    costs = pd.Series(costs)

    return (df, costs)