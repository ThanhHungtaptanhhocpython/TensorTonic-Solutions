import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    n_samples, n_features = X.shape

    # 1. Parameter initialization
    w = np.zeros(n_features)
    b = 0.0

    # 2. GD loop
    for _ in range(steps):
        z = np.dot(X,w) + b
        p = _sigmoid(z)

        errors = p - y
    # 3. Calculate Gradient Descent
        dw = (1 / n_samples) * np.dot(X.T, errors)
        db = (1 / n_samples) * np.sum(errors)

    # 4. Update Parameter

        w -= lr * dw
        b -= lr * db

    return w,b
        
    