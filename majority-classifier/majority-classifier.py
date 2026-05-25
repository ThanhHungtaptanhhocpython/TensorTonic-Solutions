import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # Write code here
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)


    labels, counts = np.unique(y_train, return_counts=True)
    
    majority_label = labels[np.argmax(counts)]
    
    n_samples = X_test.shape[0] if X_test.ndim > 1 else len(X_test)
    
    predictions = np.full(n_samples, majority_label)

    return predictions
