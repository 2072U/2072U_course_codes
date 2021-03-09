# Polynomial interpolation function, programmed in class (lecture 12) by L. van Veen and all those present.
# Contains:
# * ForwardSub (for solving a lower triangular linear system)
# * interpolate (computes the polynomial coefficients)
# * evaluate (evaluate the interpolant)
# Ontario Tech U, CSCI/MATH2072U, 2021.
import numpy as np

def ForwardSub(L,y):
    # Solve the triangular system L z = y, where L is lower triangular.
    n = np.shape(L)[0]
    z = np.zeros((n,1))
    for i in range(1,n+1):
        z[i-1] = y[i-1]
        for j in range(1,i):
            z[i-1] -= L[i-1,j-1] * z[j-1]
        z[i-1] /= L[i-1,i-1]
    return z

# Polynomial interpolation using the basis {1,(x-x0),(x-x0)(x-x1),...}. Returns the coefficients of these basis functions.
# Input: np arrays of size (n+1,1) of x and y-values.
# Output: np array of size (n+1,1) of coefficients of the functions phi (see lecture 12).
def interpolate(x,y):
    n = np.shape(x)[0] - 1
    # Define the matrix A:
    A = np.zeros((n+1,n+1))
    A[0,0] = 1.0
    for i in range(1,n+1):
        A[i,0] = 1.0
        for j in range(1,i+1):
            A[i,j] = A[i,j-1] * (x[i] - x[j-1])
    # Solve A a = y (lower triangular system)
    a = ForwardSub(A,y)
    return a

# Evaluate the interpolating polynomial at point x.
# Input: np arrays of shape (n+1,1) with interpolation nodes and coefficients; float x, the point at which the interpolant is evaluated.
# Output: value of the interpolant at x (float).
# Note, that we use the "telescope" form of the polynomial in order to evaluate it using O(n) FLOPs.
def evaluate(xs,a,x):
    n = np.shape(xs)[0] - 1
    P = a[n] * (x-xs[n-1]) + a[n-1]
    for k in range(n-1,0,-1):
        P = P * (x-xs[k-1]) + a[k-1]
    return P
