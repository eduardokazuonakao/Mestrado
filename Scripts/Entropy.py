import numpy as np
import os

def entropy(y,clustering):
#let i be a class and j a cluster (in our case i == j)

    #m = total number of data points
    m = y.size
    #print('m =  ', m)
    
    #m2[j] = number of values in cluster j
    m2 = np.zeros(len(set(y)))
    
    #m1[i][j] = number of values of class i in cluster j
    m1 = np.zeros((len(set(y)), len(set(y))))
    
    for index in range(0,m):
        m2[clustering[index]] += 1
        m1[clustering[index], y[index]] += 1
    #print('m2 = ', m2)
    #print('m1 = ', m1)

    #p[i][j] = m1[i][j]/m2[j]
    p = np.zeros((len(set(y)), len(set(y))))
    for index1 in set(y):
        for index2 in set(y):
            p[index1, index2] = m1[index2, index1] / m2[index2]
    #print('p = ', p)
    
    #e[j] = e[j] + p[i][j]*log2(p[i][j])
    e = np.zeros(len(set(y)))
    
    #totalEntropy = totalEntropy + m2[j]*e[j]/m
    totalEntropy = 0
    
    for index1 in set(y):
        for index2 in set(y):
            if p[index2][index1] != 0:
                e[index1] += p[index2, index1] * np.log2(p[index2, index1])
        #e[index1] = -e[index1]
        totalEntropy += m2[index1] * e[index1] / m
    #print('e = ', e)
    #print('totalEntropy = ', totalEntropy)
    
    return totalEntropy


entropy_GMMX = []
entropy_GMMPCA = []
entropy_GMMIsomap = []
entropy_GMMLLE = []

entropy_kmeansX = []
entropy_kmeansPCA = []
entropy_kmeansIsomap = []
entropy_kmeansLLE = []

y = np.load('E:/Eduardo/Tese/Implementação/Botswana/Botswana Matrizes/Botswana_reducedY.npy')

for i in range (1,41):
    
    for j in range (1,41):
        
        GMMXy = np.load('E:/Eduardo/Tese/Implementação/Botswana/Botswana Matrizes/GMM/Original/Reduction'+str(i)+'/Not matched/Botswana_GMMX_reduction'+str(i)+'_clustering'+str(j)+'.npy')
        GMMPCAy = np.load('E:/Eduardo/Tese/Implementação/Botswana/Botswana Matrizes/GMM/PCA/Reduction'+str(i)+'/Not matched/Botswana_GMMPCA_reduction'+str(i)+'_clustering'+str(j)+'.npy')
        GMMIsomapy = np.load('E:/Eduardo/Tese/Implementação/Botswana/Botswana Matrizes/GMM/Isomap/Reduction'+str(i)+'/Not matched/Botswana_GMMIsomap_reduction'+str(i)+'_clustering'+str(j)+'.npy')
        GMMLLEy = np.load('E:/Eduardo/Tese/Implementação/Botswana/Botswana Matrizes/GMM/LLE/Reduction'+str(i)+'/Not matched/Botswana_GMMLLE_reduction'+str(i)+'_clustering'+str(j)+'.npy')
        
        kmeansXy = np.load('E:/Eduardo/Tese/Implementação/Botswana/Botswana Matrizes/KMeans/Original/Reduction'+str(i)+'/Not matched/Botswana_kmeansX_reduction'+str(i)+'_clustering'+str(j)+'.npy')
        kmeansPCAy = np.load('E:/Eduardo/Tese/Implementação/Botswana/Botswana Matrizes/KMeans/PCA/Reduction'+str(i)+'/Not matched/Botswana_kmeansPCA_reduction'+str(i)+'_clustering'+str(j)+'.npy')
        kmeansIsomapy = np.load('E:/Eduardo/Tese/Implementação/Botswana/Botswana Matrizes/KMeans/Isomap/Reduction'+str(i)+'/Not matched/Botswana_kmeansIsomap_reduction'+str(i)+'_clustering'+str(j)+'.npy')
        kmeansLLEy = np.load('E:/Eduardo/Tese/Implementação/Botswana/Botswana Matrizes/KMeans/LLE/Reduction'+str(i)+'/Not matched/Botswana_kmeansLLE_reduction'+str(i)+'_clustering'+str(j)+'.npy')
        
        entropy_GMMX.append(entropy(y, GMMXy))
        entropy_GMMPCA.append(entropy(y, GMMPCAy))
        entropy_GMMIsomap.append(entropy(y, GMMIsomapy))
        entropy_GMMLLE.append(entropy(y, GMMLLEy))
        
        entropy_kmeansX.append(entropy(y, kmeansXy))
        entropy_kmeansPCA.append(entropy(y, kmeansPCAy))
        entropy_kmeansIsomap.append(entropy(y, kmeansIsomapy))
        entropy_kmeansLLE.append(entropy(y, kmeansLLEy))

entropy_GMMX = np.array(entropy_GMMX)
entropy_GMMPCA = np.array(entropy_GMMPCA)
entropy_GMMIsomap = np.array(entropy_GMMIsomap)
entropy_GMMLLE = np.array(entropy_GMMLLE)

entropy_kmeansX = np.array(entropy_kmeansX)
entropy_kmeansPCA = np.array(entropy_kmeansPCA)
entropy_kmeansIsomap = np.array(entropy_kmeansIsomap)
entropy_kmeansLLE = np.array(entropy_kmeansLLE)

result_directory = 'E:/Eduardo/Tese/Implementação/Botswana/Entropy'
if not os.path.exists(result_directory):
    os.makedirs(result_directory)
    
np.save(result_directory+'/Botswana_entropy_GMMX',entropy_GMMX)
np.save(result_directory+'/Botswana_entropy_GMMPCA',entropy_GMMPCA)
np.save(result_directory+'/Botswana_entropy_GMMIsomap',entropy_GMMIsomap)
np.save(result_directory+'/Botswana_entropy_GMMLLE',entropy_GMMLLE)

np.save(result_directory+'/Botswana_entropy_kmeansX',entropy_kmeansX)
np.save(result_directory+'/Botswana_entropy_kmeansPCA',entropy_kmeansPCA)
np.save(result_directory+'/Botswana_entropy_kmeansIsomap',entropy_kmeansIsomap)
np.save(result_directory+'/Botswana_entropy_kmeansLLE',entropy_kmeansLLE)

entropy_X = []
entropy_PCA = []
entropy_Isomap = []
entropy_LLE = []

entropy_X = np.array(entropy_X)
entropy_PCA = np.array(entropy_PCA)
entropy_Isomap = np.array(entropy_Isomap)
entropy_LLE = np.array(entropy_LLE)

entropy_X = np.concatenate((entropy_kmeansX,entropy_GMMX))
np.save(result_directory+'/Botswana_entropy_X',entropy_X)

entropy_PCA = np.concatenate((entropy_kmeansPCA, entropy_GMMPCA))
np.save(result_directory+'/Botswana_entropy_PCA',entropy_PCA)

entropy_Isomap = np.concatenate((entropy_kmeansIsomap, entropy_GMMIsomap))
np.save(result_directory+'/Botswana_entropy_Isomap',entropy_Isomap)

entropy_LLE = np.concatenate((entropy_kmeansLLE, entropy_GMMLLE))
np.save(result_directory+'/Botswana_entropy_LLE',entropy_LLE)