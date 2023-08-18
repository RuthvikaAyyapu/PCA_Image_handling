from functools import partial
import numpy as np
import math
import matplotlib.pyplot as plt
M = 250 #precission number for calculating inverse values.

#General inverse calculator:
#args: cdf function to be inverted, minL: the minimum value of domain (to be returned for y=0)
#maxL: the maximum value of the domain (to be returnd for y=1)
#y: y = cdfpx(t)
def cdfpxinv(cdfpx, minl,maxl,y):
    x = minl
    for n in range(0, M):
        x = x + (maxl - minl)/M
        if(cdfpx(x)>y):
            return min(x, maxl)



#pdf(x) for Triangle
def pxTri(x):
    if(x<0):
        return 0
    if(x>math.pi):
        return 0
    if(x<math.pi/3):
        return (6*x)/(math.pi**2)
    if(x<math.pi):
        return (3/math.pi)*(1 - (x/math.pi))

#cdf(x) for Triangle
def cdfpxTri(x):
    if(x>math.pi):
        return 1
    if(x<0):
        return 0
    if(x<math.pi/3):
        return (3*(x**2))/(math.pi**2)
    if(x<math.pi):
        return (2/math.pi)*(1.5*(x - (x**2)/(2*math.pi)) - math.pi/4)


#similar function to get Points in triangle.
def NpointsTri(N, cdfpx, minl, maxl):
    randvals = np.random.random(N)
    invf = partial(cdfpxinv, cdfpx, minl, maxl)
    xvals = np.vectorize(invf)(randvals)
    lvals = np.vectorize(pxTri)(xvals)
    yvals = np.random.random(N)*(lvals)
    return xvals, yvals
N = 10000000

xvals, yvals = NpointsTri(N, cdfpxTri, 0, math.pi)
#the large number of points results in inf.
#we check and exclude infinities.
finx = np.isfinite(xvals)
finy = np.isfinite(yvals)
finboth = np.vectorize(lambda x,y: x and y)(finx, finy)
print(finboth.shape)
plt.hist2d(xvals[finboth], yvals[finboth], [1000, 1000])
plt.ylim(0, 0.6)
plt.xlim(0, math.pi)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('x')
plt.ylabel('y')
plt.show()