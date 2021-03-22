import numpy as np
import matplotlib.pyplot as plt

# Test function and its derivative:
def f(x):
    return np.exp(np.sin(x))
def df(x):
    return np.cos(x) * np.exp(np.sin(x))

# Set the domain and initial number of grid points:
l = 0.0
r = np.pi
n = 4    # grid points are labelled 0 .. n
# Initialize a list of erros for plotting:
errs = np.empty((0,2))

# Loop over increasingly fine grids:
for k in range(12):
    xs = np.linspace(l,r,n+1) # Set up the grid: xs[i] = l + i * h  (i=0..n)
    h = (r-l)/float(n)        # Grid spacing
    ys = f(xs)                # Find function and its derivative on the grid
    dys = df(xs)
    fddys = np.zeros(n+1)     # Compute the finite difference approximation
    for i in range(1,n):      # on the interior grid points
        fddys[i] = (ys[i+1]-ys[i-1])/(2.0*h)
    errs = np.append(errs,[[h,np.linalg.norm(dys[1:n] - fddys[1:n],np.inf)]],0) # Remember the error - use the inf norm so the number of grid points does not matter
    n = 2*n                   # Double the number of grid points for next test

print(errs)
# Show the errors on a log scale along with O(h) and O(h^2) behaviour:
plt.loglog(errs[:,0],errs[:,1],'-*')
plt.loglog(errs[:,0],errs[:,0]**2,'-r')
plt.loglog(errs[:,0],errs[:,0],'-r')
plt.show()

# Homework: add a finite difference approximation to the derivative at grid points 0 and n while keeping the error O(h^2)!


