import h5py
import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt
from PCA import pcaVects, getcovar

#function to rescale (80, 80, 3) to [0,1]
def fitforImg(arr):
    max1 = np.max(arr)
    min1 = np.min(arr)
    arr = (arr - min1)/(max1 - min1)
    return arr



if __name__=='__main__':
    N = 16
    M = 80*80*3
    #array to read images into: using float16 to minimze operation time.
    data = np.zeros((N, 80, 80, 3), dtype=np.float16)
    for i in range(1, 17):
        data[i-1] = plt.imread(f'../data/data_fruit/image_{i}.png')
    pcadata = np.reshape(data, (N,M))
    
    ############Getting the mean
    mean = np.zeros((1, M), dtype=np.float16)
    def addtomean(x):
        global mean
        mean = mean + x
    np.apply_along_axis(addtomean, 1, pcadata)
    mean = mean/N
    print(mean.shape)
    plt.imsave('q6mean.png', mean.reshape(80,80,3))
    #########################

    #Inbuilt function for faster computation:
    # covmat = ((pcadata.shape[0]-1)/pcadata.shape[0])*np.cov(pcadata.transpose())
    #Written logic is same as the idea in PCA.py
    stan = np.apply_along_axis(lambda x: x- mean.transpose(), 1, pcadata)
    indmat = np.indices((M,M))
    covmat = np.apply_along_axis(lambda x: getcovar(stan[:,x[0]], stan[:,x[1]]), axis=0, arr=indmat)
    print(covmat.shape)
    covmat = covmat.astype('float16')
    #Calculating and cleaning eigenvalues and eigenvectors
    egvals, egvects = sl.eigh(covmat, eigvals=(M-10, M-1))
    egvects = egvects.transpose()
    ##remove complex part
    egvals = np.real(egvals)
    #sort in descending order
    sorter = egvals.argsort()
    egvals = egvals[sorter[::-1]]
    egvects = egvects[sorter[::-1]]
    top4vects = egvects[:4]

    #save top 4 principle modes of variation.
    top4vects = np.reshape(top4vects, (4, 80, 80, 3))
    top4vects = fitforImg(top4vects)
    for i in range(4):
        plt.imsave(f'q6topvect{i}.png',top4vects[i])

    #top 10 eigenvalues.
    top10vals = egvals[:10]
    np.savetxt('q6eigenvalues.csv', top10vals)
    plt.bar(np.arange(10), top10vals)
    plt.show()

    #############Saving top4vects for Q6 part 3
    top4vects = top4vects.reshape(4, 19200)
    print("saving vects")
