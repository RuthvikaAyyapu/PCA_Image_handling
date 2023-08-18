import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
def plotNum(vals_sub,vals,vals_add, label):

    fig, axes = plt.subplots(nrows=1, ncols=3)
    #fix minimum and maximum values of heatmap
    vmin = np.min(vals_sub)
    vmax = np.max(vals_add)
    #data has to be transposed, else numbers are flipped diagonally.
    im1 = axes[0].imshow(vals_sub.transpose(), cmap='hot', vmin=vmin, vmax=vmax,interpolation='nearest')
    axes[0].set_xlabel('Mean - sqrt(l)v')
 
    axes[1].imshow(vals.transpose(), cmap='hot', vmin=vmin, vmax=vmax,interpolation='nearest')
    axes[1].set_xlabel('Mean')
 
    axes[2].imshow(vals_add.transpose(), cmap='hot', vmin=vmin, vmax=vmax,interpolation='nearest')
    axes[2].set_xlabel('Mean + sqrt(l)v')
 
    #position the colorbar.
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im1, cax=cbar_ax)
    #savefigure.
    plt.savefig(f'three_images{label}.png')
    #commented code to generate heatmap on same scale for principle mode of variation.
    # egvals = np.loadtxt('eigenvalues.csv')
    # eigenvect = np.loadtxt(f'eigenvect_{int(label)}.csv')
    # plt.clf()
    # plt.imshow(sqrt(egvals[int(label)])*eigenvect.reshape(28,28).transpose(), cmap='hot', vmin=vmin, vmax=vmax, interpolation='nearest')
    # plt.colorbar()
    # plt.show()
 
if __name__=='__main__':
    for i in range(10):
        #read files.
        vals_sub = np.loadtxt(f'mean_sub_img_{i}.csv')
        vals = np.loadtxt(f'mean_img_{i}.csv')
        vals_add = np.loadtxt(f'mean_add_img_{i}.csv')
        plotNum(vals_sub,vals,vals_add, label=i)