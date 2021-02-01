# LU decomposition without any pivoting (for educational purposes only!).
# Programmed by L. van Veen and all those present in lecture 6 of CSCI/MATH20272U.
# Pseudocode on slide 33/34 in Lec_05.pdf on Canvas.
# Onterio Tech U, 2021.
import numpy as np

# Input: square array of floats A. Output: square arrays of floats L and U so that A=LU.
# Will fail by "division by zero" if (and only if) any leading pricipal sub matrices are singular. 
def LU(A):
    n = np.shape(A)[1]                           # extract number of rows/columns
    U = np.copy(A)                               # copy contents of A (avoid "U=A" to avoid changing A)
    L = np.identity(n)                           # initialize L as the identoty matrix
    for j in range(0,n-1):                       # loop over columns, excluding the last
        for i in range(j+1,n):                   # loop over column elements below the pivot
            L[i,j] = U[i,j]/U[j,j]               # compute and store the multiplier
            for k in range(j,n):                 # loop over row elements below and to the right of the pivot
                U[i,k] = U[i,k] - L[i,j]*U[j,k]  # Gauss elimination
    return L,U

            
    
    
