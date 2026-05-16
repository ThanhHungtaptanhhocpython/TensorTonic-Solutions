import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    x = np.asrray(x, dtype=float)
    results = np.where(x >=0, 1/(1+np.exp(-x)), np.exp(x)/(1 + np.exp(x)))
    return results