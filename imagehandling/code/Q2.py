import numpy as np
from numpy import random
import matplotlib.pyplot as plt 
#imported the graphic library matplotlib for plotting boxplot

mean_vector = np.array([1,2])

#initialising covariance matrix as cov_matrix
p = np.eye(2,2)
cov_matrix = p.astype(np.float64)
cov_matrix = [[1.6250, np.negative(1.9486)],[np.negative(1.9486),3.8750]]

##calculating the matrix A where C(covariance matrix)=A.A.T
#we can calculate and find out the matrix A.
A= np.empty((2,2))
A[0][0]= np.sqrt(1.625)
A[0][1]= 0
A[1][0]= np.negative(1.9486)/A[0][0]
A[1][1]=np.sqrt(3.875-A[1][0]*A[1][0])

#calculating inverse of covariance matrix.
inv_cov_matrix =np.linalg.inv(cov_matrix)
#random print statements for checking .
#print(cov_matrix)
#print(inv_cov_matrix.T)

#initialising 10 lists 5 for mean vector error(each for different value of N) and 
#5 for covaraiance matrix error(each for different value of N)
#gi corresponds to mean vector error for ith value of N
#Ci corresponds to m covaraiance matrix error for ith value of N
g1=[]
C1=[]
g2=[]
C2=[]
g3=[]
C3=[]
g4=[]
C4=[]
g5=[]
C5=[]


#coming to the main code.
for i in range(5):
    #mean_error is the list which stores the 100 mean-error values calculated below.
    mean_error=[]
    #Cn_error is the list which stores the 100 cov-error values calculated below.
    Cn_error =[] 
    #defining N as powers of 10
    N = 10**(i+1)
    #estimate covariance matrix initialising as covn_matrix
    covn_matrix = np.empty((2,N))
    u = np.empty(2)
    u =u.T

    for l in range(100):

        #calculating covn_matrix and updating u to some of all X (where X=AW+μ)
        for k in range(N):

            #taking 2 independent gaussian inputs.
            w=random.randn(2)
            x=np.matmul(A,w) + mean_vector
            covn_matrix[0][k]=x[0]-1
            covn_matrix[1][k]=x[1]-2
            u = u + x.T 

        #Data_matrix is a 2cross2 matrix obtained by multiplying 2crossN with Ncross2
        Data_matrix = np.matmul(covn_matrix,covn_matrix.T)

        #our estimate covatriance matrix is Cn .
        Cn = np.copy(Data_matrix/N)

        #Cr is the covaraiance matrix error for one instance over 100 of them.
        Cr = np.linalg.norm(cov_matrix-Cn)/np.linalg.norm(cov_matrix) 

        # appending all of them one by one to Cn_error list
        Cn_error.append(Cr)

        #mean estimate found.
        u =u/N
        u = u - mean_vector.T
        mr=np.linalg.norm(u)/np.linalg.norm(mean_vector)
        #calculated the mean error for one instance

        #appending the mean errors to mean_error list
        mean_error.append(mr)
    
    #converting both the error lists to arrays
    mean_error =np.array(mean_error)
    Cn_error = np.array(Cn_error)
    
    #now for each N we get different lists by the following if else loops we can store the mean_error and Cn_error .
    if i==0:
        g1=np.copy(mean_error)
        C1=np.copy(Cn_error)
    elif i==1:
        g2=np.copy(mean_error)
        C2=np.copy(Cn_error)
    elif i==2:
        g3=np.copy(mean_error)
        C3=np.copy(Cn_error)
    elif i==3:
        g4=np.copy(mean_error)
        C4=np.copy(Cn_error)
    elif i==4:
        g5=np.copy(mean_error)
        C5=np.copy(Cn_error)

#plotting the box plots
#plt.boxplot([g1,g2,g3,g4,g5])
plt.boxplot([C1,C2,C3,C4,C5])
plt.show()


