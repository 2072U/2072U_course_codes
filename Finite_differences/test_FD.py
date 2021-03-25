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
for k in range(21):
    xs = np.linspace(l,r,n+1) # Set up the grid: xs[i] = l + i * h  (i=0..n)
    h = (r-l)/float(n)        # Grid spacing
    ys = f(xs)                # Find function and its derivative on the grid
    dys = df(xs)
    fddys = np.zeros(n+1)     # Compute the finite difference approximation
    for i in range(1,n):      # on the interior grid points
        fddys[i] = (ys[i+1]-ys[i-1])/(2.0*h)
    # Illustration of cancellation error: two functions values almost coincide and we take the difference:
    if n==2**20:
        print('h=%e ys[1000]=%14.10f ys[1002]=%14.10f' % (h,ys[1000],ys[1002]))
    # One-sided approximation at the boundaries (we derived these to have O(h^2) error!)
    fddys[0] = (-3.0 * ys[0] + 4.0 * ys[1] - ys[2])/(2.0*h)
    fddys[n] = (3.0 * ys[n] - 4.0 * ys[n-1] + ys[n-2])/(2.0*h)

    errs = np.append(errs,[[h,np.linalg.norm(dys - fddys,np.inf)]],0) # Remember the error - use the inf norm so the number of grid points does not matter
    n = 2*n                   # Double the number of grid points for next test

print(errs)
# Show the errors on a log scale along with O(h) and O(h^2) behaviour:
plt.loglog(errs[:,0],errs[:,1],'-*')
plt.loglog(errs[:,0],errs[:,0]**2,'-r')
plt.loglog(errs[:,0],errs[:,0],'-r')
plt.show()
# Note the typical V-shape: an h too large give a large error of approximation (of the generating function by a polynomial) but an h too small give cancellation error. There is an optimal value for h and you will need to find/estimate it for every problem!
