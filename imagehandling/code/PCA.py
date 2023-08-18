import matplotlib.pyplot as plt
# import h5py
import math
import numpy as np

def getcovar(x, y):
    return np.sum(x*y)/len(x)
def pcaVects(S, redndim):
    n = S.shape[0]
    m = S.shape[1]
    #initialize a zero matrix of size m x m
    C = np.zeros((m,m))
    #get a matrix of indices: a_ij = (i, j)
    indmat = np.indices((m,m))
    #Get mean matrix = [meanx, meany]. The function q3pcaVects uses a userdefined function for this.
    means = np.mean(S, axis=0)

    #Standardize the sample.
    S = np.apply_along_axis(lambda x:x-means, axis=1, arr=S)
    #get the covariance matrix.
    C = np.apply_along_axis(lambda x: getcovar(S[:,x[0]], S[:,x[1]]), axis=0, arr=indmat)
    
    #Retrive the eigenvalues and vectors using library function.
    ans = np.linalg.eig(C)
    ans = list(ans)

    #consider only non-negative eigenvalues. The negative ones a result of numpy's brute force
    #calculations.
    negbool = np.where(ans[0]<0)
    ans[0][negbool] = 0
    
    #numpy returns eigvects as columns vects. need to transpose them.
    ans[1] = ans[1].transpose()
    
    #get the ordering of eigen values.
    sorterinds = ans[0].argsort()

    #sort eigenvectors in descending order and ignore imaginary parts.
    evects = ans[1][sorterinds[::-1]]
    evects = np.real(evects)
    
    #sort the eigenvalues.
    egvals = ans[0][sorterinds[::-1]]

    #get the demanded number of Principle components.
    princ_comps = evects[:redndim,:]

    #Get the projection matrix, P.
    P = np.matmul(S, princ_comps.transpose())
    
    #convert scalar projections into vectors.
    V = np.matmul(P, princ_comps) + means

    return P, princ_comps, egvals, V



def getSampleMean(S):
    means = np.zeros((S.shape[1],))
    for i in range(S.shape[1]):
        means[i] = np.sum(S[:,i])/S.shape[0]
    # print(means)
    # print(np.mean(S, axis=0))
    return means

###########################
#The function below follows the same logic as pcaVects, but includes user-defined functions 
#to stick to the instructions about not using certain library functions.
#Please read through the documentation for pcaVects(S, redndim) above this.
###########################
def q3pcaVects(S, redndim):
    n = S.shape[0]
    m = S.shape[1]
    C = np.zeros((m,m))
    indmat = np.indices((m,m))
    means = getSampleMean(S)
    S = np.apply_along_axis(lambda x:x-means, axis=1, arr=S)
    C = np.apply_along_axis(lambda x: getcovar(S[:,x[0]], S[:,x[1]]), axis=0, arr=indmat)
    ans = np.linalg.eig(C)
    ans = list(ans)
    #consider only non-negative eigenvalues. The negative ones a result of numpy's brute force
    #calculations.
    negbool = np.where(ans[0]<0)
    ans[0][negbool] = 0
    ans[1] = ans[1].transpose()
    sorterinds = ans[0].argsort()
    evects = ans[1][sorterinds[::-1]]
    evects = np.real(evects)
    egvals = ans[0][sorterinds[::-1]]
    princ_comps = evects[:redndim,:]
    P = np.matmul(S, princ_comps.transpose())
    V = np.matmul(P, princ_comps) + means
    #       nxr,        rxm,        rx1     nxm
    return P, princ_comps, egvals, V

