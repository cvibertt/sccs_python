import numpy as np

def Yproduct(S, M):
    """
    Create a matrix by pointwise multiplying a matrix M by each column of matrix S.
    
    Parameters:
    S : array-like
        Matrix S.
    M : array-like
        Matrix M.
    
    Returns:
    product : ndarray
        Resulting matrix.
    """
    S = np.array(S)
    M = np.array(M)
    
    product = np.full((S.shape[0], S.shape[1] * M.shape[1]), np.nan)
    
    for i in range(S.shape[1]):
        start = M.shape[1] * i
        end = M.shape[1] * (i + 1)
        product[:, start:end] = S[:, i:i+1] * M
    
    return product