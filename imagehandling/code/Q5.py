import sys
import matplotlib.pyplot as plt
import h5py
import numpy as np
from numpy.linalg.linalg import eig, eigvals
from PCA import pcaVects, getcovar
def plotBoth(orig, recon, label):
    #commented code to use csv files.
    # orig = np.loadtxt(f'orig_{label}.csv')
    # recon = np.loadtxt(f'recon_{label}.csv')
    
    #fix minimum and maximum values of heatmap
    vmin = np.min(orig)
    vmax = np.max(orig)
    plt.subplot(121)
    plt.imshow(orig.transpose(), cmap='hot',interpolation='nearest')
    plt.xlabel('Original')
    plt.colorbar()
    plt.subplot(122)
    #clip the values of the reconstructed image.
    #(The values outside the range (vmin, vmax) are mathematical anomalies and can be
    # safely ignored.)
    #If they are considered for the colormap, the images are tougher to compare. 
    plt.imshow(recon.transpose(), cmap='hot',vmin=vmin, vmax=vmax,interpolation='nearest')
    plt.xlabel('Reconstructed')
    plt.colorbar()
    # plt.show()
    plt.savefig(f'orig_recon_{label}.png')



filepath = '../data/mnist.mat'
arrays = {}
f = h5py.File(filepath)
for k, v in f.items():
    arrays[k] = np.array(v)
['digits_test', 'digits_train', 'labels_test', 'labels_train']
labels = arrays['labels_train'][0]
data = arrays['digits_train']
ardata = []
digs, freqs = np.unique(labels, return_counts=True)
print(digs, freqs)
for dig in digs:
    ardata.append(data[np.where(labels==dig)])
for i in range(len(ardata[:])):
    dig = ardata[i]
    print(dig.shape)
    meanmat = np.mean(dig, axis=0)
    l = len(dig)
    testarr = np.array(dig)
    testarr = np.reshape(testarr, (l, 784))
    
    #number of new dimensions.
    thres = 84
    #userdefined pca function, please see PCA.py
    #the 84 coordinates are 'eigenvectors'
    P, eigenvectors, egvals, V=pcaVects(testarr, thres)    
    #The 84 coordinates are stored in the projection matrix P.
    #Pick first image in each digit's data set for reconstruction:
    original = dig[0]

    #Reconstruct based on algorithm defined in report.
    #Reconstructing only first instance in data set for each digit
    reconstructed = np.matmul(P[0].reshape(1,84), eigenvectors).reshape(28, 28) + meanmat
    #Save data to avoid spending time here again.
    np.savetxt(f'recon_{i}.csv', reconstructed)
    np.savetxt(f'orig_{i}.csv', original)
    #save image comparing reconstructions.
    plotBoth(original, reconstructed, i)    
