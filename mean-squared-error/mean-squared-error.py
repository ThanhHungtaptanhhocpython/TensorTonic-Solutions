import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Write code here
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    losses = (y_pred - y_true)**2
    
    MSE = np.mean(losses)
    return MSE
    pass
