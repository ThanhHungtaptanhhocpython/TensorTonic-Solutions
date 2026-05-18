import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.asarray(A)
    n_rows, n_cols = A.shape
    A_T = np.zeros((n_cols,n_rows))
    for i in range(n_rows):   # # Vòng lặp ngoài duyệt qua các hàng (i = 0, 1)
        for j in range(n_cols):    # Vòng lặp trong duyệt qua các cột (j = 0, 1, 2)
            A_T[j][i] = A[i][j]
    return A_T
    
