# Script to illustrate the use of least-squares approximation for fitting noisy data.
# By L. van Veen, OnTechU, 2021.
import numpy as np
import matplotlib.pyplot as plt
from interpolate import *         # This is the interpolation function from the course_codes repository.

# First, we do polynomial interpolation for data that fall on a straight line:
xs = np.array([0.4,0.9,1.7,1.8,2.3,2.6])
def lin(x):
    return 2.0*x - 1.5
ys = lin(xs)
n = len(xs) - 1         # n is the maximal order of interpolation.

a = interpolate(xs,ys)  # Find the coefficients of the interpolating polynomial. Note, that only the first two are non-zero since the data fall on a straight line.
print(a)

# Plot the interpolant and the data.
xp = np.linspace(xs[0]-0.5,xs[n]+0.5,1000)
yp = evaluate(xs,a,xp)
plt.plot(xs,ys,'*')
plt.plot(xp,yp,'-k')
plt.show()

# now we add some noise to the data. Change epsilon to set the noise level.
z = np.random.rand(n+1)-0.5
eps = 1e-1
ys = ys + eps * z
# Interpolate again, now the data no longer fall on a straight line.
a = interpolate(xs,ys)

# Show the result:
xp = np.linspace(xs[0]-0.1,xs[n]+0.1,1000)
yp = evaluate(xs,a,xp)
plt.plot(xs,ys,'*')
plt.plot(xp,yp,'-k')
plt.show()

# The solution: a linear leas-squares fit. See the slides of lecture 14.
V = np.ones((n+1,2))
V[:,1] = xs.T
W = V.T @ V
R = V.T @ ys.T
b = np.linalg.solve(W,R) # Solve the 2X2 linear system (the "normal equations")
print(b)
# Note, that the slope and offset are approximately those of the original function "lin"!
def LS(x):
    return b[0]+b[1]*x
yp = LS(xp)
# Plot the result:
plt.plot(xs,ys,'*')
plt.plot(xp,yp,'-k')
plt.show()
