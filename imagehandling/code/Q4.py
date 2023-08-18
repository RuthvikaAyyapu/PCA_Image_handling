import matplotlib.pyplot as plt
import h5py
import numpy as np
from math import sqrt
from PCA import pcaVects, getcovar
###############################################
#Reading from .mat file.
filepath = '../data/mnist.mat'
arrays = {}
f = h5py.File(filepath)
for k, v in f.items():
    arrays[k] = np.array(v)
['digits_test', 'digits_train', 'labels_test', 'labels_train']
labels = arrays['labels_train'][0]
data = arrays['digits_train']
###############################################

####Segragating images based on labels.
digs = np.unique(labels)
#
#ardata holds the numpy array of images for digits 0 to 9
ardata = []
#eigenvalues collects eigenvalues for each digit.
eigenvalues = []

###Collecting all images digitwise into ardata:
for dig in digs:
    ardata.append(data[np.where(labels==dig)])

#for each digit's data set in ardata:
for i in range(len(ardata)):
    l = len(ardata[i])
    m = 784
    ##reshape:
    reshaped = np.reshape(ardata[i], (l, 28*28))
    
    #initialize zero array for means.
    mean = np.zeros(784)

    #define function to add to local array mean.
    def addtomean(x):
        global mean
        mean = mean + x
    #apply addtomean to inner images of reshaped dataset
    np.apply_along_axis(addtomean, 1, reshaped)

    #divide mean by sample size:
    mean = mean/l
    #save mean as 1x784 matrix.
    np.savetxt(f'mean_{i}.csv', mean)

    #standardize reshaped
    stan_reshaped = np.apply_along_axis(lambda x:x-mean, axis=1, arr=reshaped)
    

    #procedure similar to pcaVect from PCA.py. Please refer to that function in PCA.py.
    indmat = np.indices((m,m))
    covmat = np.apply_along_axis(lambda x: getcovar(stan_reshaped[:,x[0]], stan_reshaped[:,x[1]]), axis=0, arr=indmat)
    
    np.savetxt(f'covmat_{i}.csv', covmat)
    
    
    egvals, egvects = np.linalg.eig(covmat)
    egvects = egvects.transpose()
    negbool = np.where(egvals<0)
    egvals[negbool] = 0
    egvals = np.real(egvals)
    egvects = np.real(egvects)
    sorterinds = egvals.argsort()
    egvects = egvects[sorterinds[::-1]]
    egvals = egvals[sorterinds[::-1]]
    
    
    plt.bar(np.arange(len(egvals)),egvals)
    plt.ylabel('Eigen Value')
    plt.show()
    plt.savefig(f'EigenValues_all_{i}.png')
    plt.clf()
    plt.bar(np.arange(len(egvals[:100])),egvals[:100])
    plt.ylabel('Eigen Value')
    plt.show()
    plt.savefig(f'EigenValues_100_{i}.png')
    
    #isolate principal mode of variation.
    mainvect = egvects[0]
    mean = np.reshape(mean, (28, 28))
    
    eigenvalues.append(egvals[0])
    data1 = mean - sqrt(egvals[0])*np.reshape(mainvect, (28,28))
    data2 = mean + sqrt(egvals[0])*np.reshape(mainvect, (28,28))
    
    ##save files to be used by 3imgs.py
    np.savetxt(f'mean_img_{i}.csv', mean)
    np.savetxt(f'mean_sub_img_{i}.csv', data1)
    np.savetxt(f'mean_add_img_{i}.csv', data2)
    np.savetxt(f'eigenvect_{i}.csv', mainvect)
np.savetxt('eigenvalues.csv', np.array(eigenvalues))