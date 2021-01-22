# An implementation of Newton iteration, written in lecture 4 of MATH/CSCI2072U Scientific Computing I.
# By L. van Veen all all present in the lecture. Ontario Tech U, 2021.
import numpy as np

def Newton(f,df,x0,tol_x,tol_r,kMax):
    x = x0
    conv = 0
    for k in range(kMax):
        fx = f(x)
        fpx = df(x)
        dx = -fx/fpx
        error_est = abs(dx)      # dx is x_{k+1} - x_{k}
        residual = abs(fx)
        x += dx
        print("it %d err=%e res=%e" % (k,error_est,residual))
        if error_est < tol_x and residual < tol_r:
            conv = 1
            break
    if conv == 0:
        print("Warning: no convergence after %d iterations." % (kMax))
    return x,error_est,residual
