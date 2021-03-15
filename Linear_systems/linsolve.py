# Auxiliary routine for solving a linear system A x = b. Calls our own LUP decomposition routine, as well as the forward and backward substitution routines, in LUP.py.
# By L. van Veen, OnTechU, 2021.
import numpy as np
from LUP import *

def linsolve(A,b):
    # Solve A x = b with LUP decomposition.
    L,U,P,par,ok = LUP(A)
    if ok == 0:
        print("Warning: nearly degenerate Jacobian!")
    y = ForwardSub(L,P,b)
    x = BackwardSub(U,y)
    return x
