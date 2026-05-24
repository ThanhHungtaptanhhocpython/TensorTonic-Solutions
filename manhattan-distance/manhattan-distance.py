import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    x = np.asarray(x)
    y = np.asarray(y)

    result = 0
    for i in range(len(x)): 
        result += abs(x[i] - y [i])

    return float(result)
        
    