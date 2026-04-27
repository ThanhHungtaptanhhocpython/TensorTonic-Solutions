import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    error = y_true - y_pred
    abs_error = np.abs(error)
    
    losses_mae = 0.5 * (error**2)
    losses_mse = delta*(abs_error - 0.5*delta)

    losses = np.where(abs_error <= delta, losses_mae, losses_mse)

    result = np.mean(losses)
    return result
    pass