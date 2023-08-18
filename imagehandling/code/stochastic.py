from math import exp, pi
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
NEG_INF = -100
POS_INF = -NEG_INF
INTEGRAL_N = 100 ###the n to use for reimann sums
#A measure of precision. Higher M gives higher precision
M = 1000
dx = (POS_INF - NEG_INF) / (M-1)
#Approach 1 to generating stochastic data
#Obtain a series of random numbers between 0 and 1.
#map them to domain via inverse function of CDF. CDF-1: [0,1]->(-inf, +inf)
def integrate(f, a, b):
    N = INTEGRAL_N
    f = np.vectorize(f)
    #numpy can now use f efficiently on its arrays
    vals = np.linspace(int(a*N), int(b*N), 1+int((b-a)*N))
    #vals is a 1-D array : [a*N, .... , b*N] with 1+(b-a)*N elements
    #Using the reiman formula for the integral:
    integral = np.sum(f(vals/N)*1/N)
    return integral
def gencdf(f, x):
    #using the formula for cdf
    return integrate(f, NEG_INF, x)
def inversefunction(f, y):
    x = NEG_INF
    dx = (POS_INF-NEG_INF)/INTEGRAL_N
    # print(y)
    while(x<POS_INF):
        if(f(x)>=y):
            return x
        x = x+ dx
    return POS_INF
def stochasticdata2(pdf, N):
    cdf = partial(gencdf,pdf)
    cdf_inverse = partial(inversefunction, cdf)
    data = np.random.random(N)
    data = np.vectorize(cdf_inverse)(data)
    return data

#Approach 2: faster function, because it uses book-keeping of calculated values.

def stochasticdata(f, n, ni=NEG_INF, pi=POS_INF, return_dx=False, precision_degree=M):
    lap = np.vectorize(f, otypes=[np.float64])
    nums = np.linspace(ni, pi, precision_degree, endpoint=False)
    lapnum = lap(nums)
    print(nums)
    I = dx*np.sum(lapnum)
    frac = lambda x: f(x)*dx/I

    a = np.vectorize(frac)
    nums:np.ndarray
    def cumA(x):
        for i in range(len(nums)):
            # print(nums[i], x)
            if(nums[i]>x):
                # print(x)
                return np.sum(a(nums[:i]))
        return 1

    cumvals = np.vectorize(cumA)((nums))
    # plt.plot(cumvals)
    def randgen(x):
        lim = cumvals[-1]
        mapped1 = (lim)*(x)
        # print(mapped1)
        for i in range(0, len(cumvals)):
            if(cumvals[i]>mapped1):
                return nums[i]
        return nums[-1]
    
    data = np.random.random(n)
    data = np.vectorize(randgen)(data)
    if(return_dx):
        return data, (pi-ni)/precision_degree
    else:
        return data