import matplotlib.pyplot as plt
import h5py
import math
import numpy as np
from numpy.core.fromnumeric import _mean_dispatcher
from numpy.core.numeric import indices
from PCA import getcovar, pcaVects, q3pcaVects
#############
#Reading from .mat file in python:
filepath = '../data/points2D_Set1.mat'
filepath2 = '../data/points2D_Set2.mat'
arrays = {}
f1 = h5py.File(filepath)
for k, v in f1.items():
    arrays[k] = np.array(v)
#############

xvals = arrays['x'][0]
yvals = arrays['y'][0]

#sample matrix: Snx2
#coverting Xnx1 and Ynx1 to Snx2
S = np.dstack((xvals.ravel(), yvals.ravel()))[0]

#Calling user-defined PCA Function:
#returns 
P,egvects,egvals,V = q3pcaVects(S, 1)

plt.scatter(xvals, yvals)
plt.plot(V[:,0], V[:,1], linewidth=3, color='orange')
plt.show()


f2 = h5py.File(filepath2)
for k, v in f2.items():
    arrays[k] = np.array(v)


xvals = arrays['x'][0]
yvals = arrays['y'][0]

#sample matrix: Snx2
#coverting Xnx1 and Ynx1 to Snx2
S = np.dstack((xvals.ravel(), yvals.ravel()))[0]

#Calling user-defined PCA Function:
#returns 
P,egvects,egvals,V = q3pcaVects(S, 1)

plt.scatter(xvals, yvals)
plt.plot(V[:,0], V[:,1], linewidth=3, color='orange')
plt.show()
