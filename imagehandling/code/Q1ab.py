from functools import partial
import numpy as np
import math
import matplotlib.pyplot as plt
M = 500 #precission number for calculating inverse values.
#returns cdf_x(t) for the ellipse part.
def cdfpxEllipse(x, a):
    if(x>a):
        return 1
    if(x<-a):
        return 0
    return (0.5  + (1/math.pi)*(math.asin(x/a) + (x/a)*math.sqrt(1 - ((x/a)**2))))


#General inverse calculator:
#args: cdf function to be inverted, minL: the minimum value of domain (to be returned for y=0)
#maxL: the maximum value of the domain (to be returnd for y=1)
#y: y = cdfpx(t)
def cdfpxinv(cdfpx, minl,maxl,y):
    x = minl
    for n in range(0, M):
        x = x + (maxl - minl)/M
        if(cdfpx(x, maxl)>y):
            return min(x, maxl)


#function to return N xvalues and yvalues
#args: a: length of axis||x, b:length of axis || to y, cdfpx: the cdf function for x
def NpointsEllipse(a, b, N, cdfpx):
    #limits of x
    minl = -a
    maxl = a
    #get N random values between 0 and 1
    randvals = np.random.random(N)
    #define cdfx-1() function
    invf = partial(cdfpxinv, cdfpx, minl, maxl)
    #get xvalues : cdfpx-1(R())
    xvals = np.vectorize(invf)(randvals)
    #get the l(x) values corresponding to collected xvalues.
    lvals = np.vectorize(lambda x: b*math.sqrt(1 - ((x/a)**2)))(xvals)

    #get yvalues l(x)*(R() - 0.5)
    yvals = (np.random.random(N)*(2*lvals)) - lvals
    return xvals, yvals

N = 10000000
a = 2;b = 1
xvals, yvals = NpointsEllipse(a,b,N, cdfpxEllipse)
plt.hist2d(xvals, yvals, [1000, 1000])
plt.ylim(-b, b)
plt.xlim(-a, a)
print(min(xvals))
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('x')
plt.ylabel('y')
plt.show()