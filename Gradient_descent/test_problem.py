# Test problem for the basic gradient descent algorithm. By L. van Veen, OnTechU, 2021.
import numpy as np
import matplotlib.pyplot as plt
from basicGD import basicGD

# This test function has a horseshoe-shaped valley with a single minimum at (1,-1):
def E(x):
    return (x[1]+x[0]**2)**2 + (x[0]-1)**2

# Set the initial point:
x0 = np.array([[-3.2],[-3.9]])
# Gradient descent parameters: initial step size, factor for comparing the decrease in E to linear behaviour,
# factor by which to increase/decrease the step size, minimum value for the step size.
d = 0.1
alpha = 0.2
beta = 2.0
thr = 0.01
# Exact solution for plotting:
ex = np.array([[1.0],[-1.0]])
print('Minimal value is %e at x=%e, y=%e' % (E(ex),ex[0],ex[1]))

# Call our function:
x,res = basicGD(E,x0,d,alpha,beta,thr)
nStep = np.shape(x)[1]
print('Result is x=%e, y=%e' % (x[0,nStep-1],x[1,nStep-1]))
print('With residual E=%e' % (res))

# Compute the objective function on a nPlot X nPlot grid to find its contours:
nPlot = 1000
xp = np.linspace(-4,4,nPlot)
yp = np.linspace(-10,4,nPlot)
z = np.zeros((nPlot,nPlot))
# Set the isolevels (multiplicative scaling looks better than additive):
le=[0.0,0.125]
for i in range(1,12):
    le.append(le[i]*2)
for i in range(nPlot):
    for j in range(nPlot):
        z[j,i] = E(np.array([[xp[i]],[yp[j]]]))

# Plot the contours of E, the exact solution and the intermediary points produced by the algorithm:
plt.contourf(xp,yp,z,levels=le)
plt.plot(x[0,:],x[1,:],'-*')
plt.plot(ex[0],ex[1],'.',markersize=20)
plt.show()


