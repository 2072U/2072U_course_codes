# Code for basic gradient descent. Step size is increased after very accpted step and reduced after every rejected step.
# The algorithm terminates when the step size drops below the threshold specified as input.
# By L. van Veen, OnTechU, 2021.
import numpy as np

# Inputs: handle E of the objective function, initial point x0, initial step size d, factor 0<= alpha <=1 that determines how much the
# objective function must decrease wrt linear behaviour to accept a step, factor beta by which d is increased/reduced, threshold for d.
# Output: list of points (for plotting) and the final value of the objective function.
# Input x0 is expected to be a numpy array of shape (n,1).
def basicGD(E,x0,d,alpha,beta,thr):
    # Initialize: set initial point, value of E, convergence flag and list of points: 
    x = np.copy(x0)
    e = E(x0)
    converged = 0
    xList = x0
    # Repeat until the step size drops below thr:
    while converged == 0:
        g = FDgrad(E,x)                     # Approximate the gradient by forward finite differences.
        normg = np.linalg.norm(g,2)         # Remember the norm of the gradient to test to tentative step.
        g = g/normg                         # Set g as the gradient descent direction.
        y = x - d * g                       # Tentative step: distance d along the gradient descent direction.
        z = E(y)                            # Sample the objective function at the new point.
        if z < e - alpha * d * normg:       # Test if E decreased enough.
            x = y                           # If so, update the position and objective function value, store
            e = z                           # the new point and increase the step size.
            xList = np.append(xList,x,1)
            d *= beta
        else:                        
            d /= beta                       # If not, reject the step and reduce the step size.
        if d < thr:
            converged = 1                   # If the step size drops below the threshold, stop.
            
    return xList,e

# Forward finite differences to approximate the gradient of E. Uses hard-coded finite difference step size (bad).
def FDgrad(E,x):
    n = np.shape(x)[0]
    g = np.zeros((n,1))
    eps = 1e-5
    e = E(x)
    for i in range(n):
        y = np.copy(x)
        y[i] += eps
        z = E(y)
        g[i] = (z-e)/eps
    return g
