import matplotlib.pyplot as plt
import numpy as np
from Q6 import fitforImg
from PCA import getcovar
from math import sqrt
M = 19200
N = 16

##Reading Stored data.
# import/ os
# print(os.getcwd())
# os.chdir('../results/Q6/')
# print(os.listdir())
top4vects = np.loadtxt('../results/Q6/q6top4vects.csv')
top10vals = np.loadtxt('../results/Q6/q6eigenvalues.csv')
#normalizing eigenvectors and multiplying eigenvalues by that value
for i in range(4):
    top4vects[i] = top4vects[i]/np.linalg.norm(top4vects[i])
    top10vals[i] = top10vals[i]*np.linalg.norm(top4vects[i])
    top10vals[i] = sqrt(top10vals[i]) #sqrt because sqrt(eigenvalue) is used from here onwards.

##Run Q6.py first part to generate q6mean.png first
mean = plt.imread('../results/Q6/q6mean.png')[:,:,:3].reshape(1,M)


#find mean in coordinate fram of 4 eigenvectors.
newmean = np.matmul(mean, top4vects.transpose())
                    #1 x M M x 4
#find new covariance matrix by projecting original data
#onto four-eigenvector span,
newcovmat = np.zeros((M, M))

data = np.zeros((N, 80, 80, 3), dtype=np.float16)
for i in range(1, 17):
    data[i-1] = plt.imread(f'../data/data_fruit/image_{i}.png')
diag_eigvals = np.diag(top10vals[:4])
pcadata = np.reshape(data, (N,M))

#projecting data:
dat_in_new_coords = np.matmul(pcadata, top4vects.transpose())

# newcovmat = ((pcadata.shape[0]-1)/pcadata.shape[0])*np.cov(dat_in_new_coords.transpose())
standard = np.apply_along_axis(lambda x: x- newmean.transpose(), axis=1, arr=dat_in_new_coords)
indmat = np.indices((4,4))
newcovmat = np.apply_along_axis(lambda x: getcovar(dat_in_new_coords[:,x[0]], dat_in_new_coords[:,x[1]]), axis=0, arr=indmat)

#factor in weight of each eigenvector.
scaledeigvects = np.matmul(diag_eigvals, top4vects)

#pull from gaussian distribution defined by new mean and newcovmat
X = np.random.multivariate_normal(newmean.reshape(4,),newcovmat, 100)
newfruit = np.matmul(X, scaledeigvects).reshape(100, 80, 80, 3)
for i in range(100):
    plt.imshow(fitforImg(newfruit[i]))
    plt.show()
