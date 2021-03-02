# Very simple and sub-optimal interpolation routine.
# Input: np column arrays with x and y values of the interpolation points.
# Output: np column array with polynomial coefficients in increasing order.
# Note: this function uses the Vandermonde matrix, which is often ill-conditioned.
# By L. van Veen and all those in lecture 11 of CSCI/MATH2072U, OnTechU, 2021.
import numpy as np
from LUP import *

def interpol(x,y):
    # Inputs: arrays of shape (n+1,1) of floats representing the interpolation data
    n = np.shape(x)[0] - 1
    # Make the Vandermonde matrix [[1,x_0,x_0^2,..,x_0^n],[1,x_1,x_1^2,...,x_1^n],...,[1,x_n,x_n^2,...,x_n^n]]
    # The double loop to compute elements of V is now O(n^3). Can you make it O(n^2)?
    V = np.ones((n+1,n+1))
    for i in range(n+1):
        for j in range(1,n+1):
            V[i,j] = x[i]**j
    # Solve V a = y (you can also do `a = np.linalg.solve(V,y)`) 
    L,U,P,par,ok = LUP(V)
    if ok==0:
        print("Warning: degenerate Vandermonde matrix!")
    z = ForwardSub(L,P,y)
    a = BackwardSub(U,z)
    return a
