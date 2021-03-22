import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.exp(np.sin(x))
def df(x):
    return np.cos(x) * np.exp(np.sin(x))

l = 0.0
r = np.pi
n = 4    # grid points are labelled 0 .. n
errs = np.empty((0,2))

for k in range(12):
    xs = np.linspace(l,r,n+1) # xs[i] = l + i * h  (i=0..n)
    h = (r-l)/float(n)
    ys = f(xs)
    dys = df(xs)
    fddys = np.zeros(n+1)
    for i in range(1,n):
        fddys[i] = (ys[i+1]-ys[i-1])/(2.0*h)
    errs = np.append(errs,[[h,np.linalg.norm(dys[1:n] - fddys[1:n],2)]],0)
    n = 2*n

print(errs)
plt.loglog(errs[:,0],errs[:,1],'-*')
plt.loglog(errs[:,0],errs[:,0]**2,'-r')
plt.loglog(errs[:,0],errs[:,0],'-r')
plt.show()


