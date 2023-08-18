import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt

from Q6 import fitforImg
###function to show mean and top 4 eigenvectors
def showMean4PMVs(mean, eigvects):
    plt.subplot(151)
    plt.imshow(mean.reshape((80, 80, 3)))
    plt.xlabel('Mean')
    for i in range(4):
        plt.subplot(152 + i)
        plt.imshow(fitforImg(eigvects[i].reshape((80, 80, 3))))
        plt.xlabel(f'Princ. Mode. {i+1}')
    plt.show()
N = 16
M = 80*80*3
####Reading data again.
data = np.zeros((N, 80, 80, 3), dtype=np.float16)
for i in range(1, 17):
    data[i-1] = plt.imread(f'../data/data_fruit/image_{i}.png')
pcadata = np.reshape(data, (N,M))

#Reading vectors from previously saved file.
eigvects = np.zeros((4, M))
for i in range(4):
    arr = plt.imread(f'../results/Q6/q6topvect{i}.png')
    arr = arr[:,:,:3].reshape((1, M))
    eigvects[i] = (arr)/np.linalg.norm(arr)
#Reading mean
mean = plt.imread('../results/Q6/q6mean.png')[:,:,:3].reshape((1, M))
showMean4PMVs(mean, eigvects)



#projecting data onto four eigenvectors to get Minimized frobenius norm
for i in range(16):
    # print(eigvects)
    projection = np.matmul(pcadata[i], eigvects.transpose())
    print(projection)
    plt.subplot(121)
    plt.imshow(fitforImg(np.matmul(projection, eigvects).reshape((80, 80, 3))+mean.reshape(80, 80, 3)))
    plt.xlabel('Projected onto 4 PMVs')
    plt.subplot(122)
    plt.imshow(plt.imread(f'../data/data_fruit/image_{i+1}.png'))
    # plt.imshow()
    plt.xlabel('Original')
    # plt.show()
    plt.savefig(f'q6_recon_{i+1}.png')
    projto4vects = np.matmul(projection, eigvects) + mean
    projto4vects = fitforImg(projto4vects.reshape(80, 80, 3))
    plt.imsave(f'recon{i+1}.png', projto4vects)
    # plt.show()