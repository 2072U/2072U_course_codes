# Author: L. van Veen, UOIT, 2021. CSCI/MATH2072U, lecture 6, slide 7/17.
# PA=LU decomposition that keeps track of the parity of the permutation (even=1, odd=-1).
#                    GNU GENERAL PUBLIC LICENSE
#                      Version 3, 29 June 2007
import numpy as np

# Note: indices i,j and k are used as matrix indeces (starting from 1) while s is used as python array index (starting from 0).
def LUP(A):
    small = 1e-12                                 # a pivot smaller than this will raise the error flag "ok=0"
    n = np.shape(A)[0]                            # extract matrix size
    U = np.copy(A)                                # copy content of A (avoid linking U and A)
    L = np.identity(n)                            # initialize L and P
    P = np.identity(n)
    par = 1                                       # initial permutation (identity) is even
    ok = 1                                        # by default, we assume the matrix is non-singular
    for k in range(1,n):
        # optional: print the intermediary steps
        print('------------ working on column %d ------------' %(k))
        print(L)
        print(U)
        print(P)

        s = np.argmax(abs(U[k-1:n,k-1]))+k-1   # find pivot element
        if abs(U[s,k-1]) < small:                 # check if pivot is too close to zero
            print("(nearly) singular matrix, pivot smaller than %e" % small)
            ok = 0
            break                                 # matrix is too close to singular, exit with error flag up
        if s != k-1:                              # if the pivot is not on the diagonal...
            par = -par                            # change parity
            U = swap(U,s,k-1,n)                   # swap rows of U
            if k>1:                               # swap rows of L left of diagonal element
                L = swap(L,s,k-1,k-1)
            P = swap(P,s,k-1,n)                   # swap rows of P
        for j in range(k+1,n+1):                  # Gauss elimination of rows below pivot
            L[j-1,k-1] = U[j-1,k-1]/U[k-1,k-1]    # store the multiplier in L
            for i in range(k,n+1):                # form linear combination of rows
                U[j-1,i-1]=U[j-1,i-1] - L[j-1,k-1]*U[k-1,i-1]
    return L,U,P,par,ok

def swap(M,i,j,k):
# Swap rows i and j from column 0 up to (but not including) k.
# Indices in this function are used as python array indices (starting from 0).
    dum = np.copy(M[i,0:k])
    M[i,0:k] = np.copy(M[j,0:k])
    M[j,0:k] = np.copy(dum)
    return M

def ForwardSub(L,P,y):
    # Solve the triangular system L z = P y, where L is lower triangular and has diagonal elements equal to 1 and P is a permutation matrix.
    n = np.shape(L)[0]
    z = np.zeros((n,1))
    y = np.dot(P,y)
    for i in range(1,n+1):
        z[i-1] = y[i-1]
        for j in range(1,i):
            z[i-1] -= L[i-1,j-1] * z[j-1]
        z[i-1] /= L[i-1,i-1]
    return z

def BackwardSub(U,z):
    # Solve the triangular system U x = z, where U is upper triangular.
    n = scipy.shape(U)[0]
    x = scipy.zeros((n,1))
    for i in range(n,0,-1):
        x[i-1] = z[i-1]
        for j in range(i+1,n+1):
            x[i-1] -= U[i-1,j-1] * x[j-1]
        x[i-1] /= U[i-1,i-1]
    return x






    
