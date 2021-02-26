# Function for Newton-Raphson iteration. Written in class by L. van Veen and all students in CSCI/MATH2072U present.

import numpy as np
from linsolve import linsolve          # This is our own module for linear solving with LUP decomposition.

# Newton-Raphson iteration. Inputs: function handles F (returns n-vector) and DF (returns nXn Jacobian), initial guess x0 (n-vector), max nr of iterations kMax, tolerance for the residual and error. Returns: approximate solution x, final error and residual.
def NR(F,DF,x0,kMax,tol_res,tol_err):
    x = x0                             # x will be updated in the loop.
    conv = 0                           # Convergence flag. 
    for k in range(kMax):              # Loop over Newton-Raphson iterations.
        r = F(x)                       # Compute the residual vector.
        J = DF(x)                      # Compute the Jacobian.
        dx = linsolve(J,-r)            # Alternatively, you can use np.linalg.solve(J,-r).
        x = x + dx                     # Apply the update.
        err = np.linalg.norm(dx,2)     # Compute the approximate error and residual.
        res = np.linalg.norm(r,2)
        print("it %d res=%e appr err=%e" % (k,res,err))
        # Test for convergence:
        if err < tol_err and res < tol_res:
            conv = 1
            break
    if conv == 0:                      # Print a warning if the iteration did not converge.
        print("Warning: no convergence!")
    return x,err,res

