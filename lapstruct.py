# python code of LAPSTRUCT version 1.0, based on R code by junzhang@galton.uchicago.edu 

import numpy as np
import pandas as pd
import scipy.linalg as la
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from scipy import sparse

def IndCov(G):
    NM=G.shape
    print(G)
    n=NM[0]
    m=NM[1]
    fre = np.zeros((m, 1)); 
    for i in range(m): 
        fre[i] = np.mean(G[:,i])/2.0; 
    print(any(fre==0))
    for i in range(m):
        for j in range(n):
            try:
                G[j,i] = (G[j,i]-2.0*fre[i])/np.sqrt(2.0 * fre[i] * (1.0-fre[i]))
            except:
                pass
                #print()
                #print(G[j,i], fre[i])
                #print(np.sqrt(2.0 * fre[i] * (1.0-fre[i])))
                #print((2.0 * fre[i] * (1.0-fre[i])))
                #print((G[j,i]-2.0*fre[i])/np.sqrt(2.0 * fre[i] * (1.0-fre[i])))
    return np.matmul(G, np.transpose(G))/m


## compute the Laplacian eigenfunctions based on the covariance matrix of samples
## parameters: (1) 1.0-eps measures the distance between each pair of sample in the sense of correlation,
##                  one usually needs tune the eps for meaningful structure, see reference for discussion.
##             (2) t is a scale tuning para which is set to 1.0 in all the computation.  
def _lapstruct(idcov, eps, t):
        NN=idcov.shape
        N=NN[0]
        w=np.zeros((N, N))
        d=np.zeros((N, N));
        L=np.zeros((N, N));
        for i in range(N): 
            for j in range(N):
                if idcov[i,j] > eps:
                    w[i,j] = np.exp(-(1.0-idcov[i,j])*(1.0-idcov[i,j]));
                else:
                    w[i,j] = 0.0; 
        for i in range(N):
            for j in range(N):
                d[i,i] = d[i,i] + w[i,j]
        L = d - w
        for i in range(N): 
            for j in range(N): 
                L[i,j] = L[i,j]/np.sqrt(d[i,i]*d[j,j]) 
        
        return la.eig(L)[1]
    
    
def lapstruct(values, eps=-0.1, t=1.0):
    return _lapstruct(IndCov(values), eps, t)


if __name__ == "__main__":
    G = pd.read_csv("SNP.csv", header=None)
    G = G.iloc[:100 , 1:]
    l = lapstruct(G.values)
    plt.scatter(l[:, -1], l[:,-2])
    plt.show()
#load the genotype data matrix. One needs change to the right directory.
#G=as.matrix(read.table("input_hgdp800_genotype.txt"))



#idcov=IndCov(G)
#Here eps is set to -0.30 for demonstration on the HGDP panel dataset for global population structure.
#lapevec = lapstruct(idcov,-0.30, 1.0)
#plot(lapevec[,N-1],lapevec[,N-2], xlab="Lap1", ylab="Lap2");


