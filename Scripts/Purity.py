import numpy as np
import os

def purity(y,clustering):
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
    
    P = np.zeros(len(set(y)))
    
    totalPurity = 0
    
    for index1 in set(y):
        for index2 in set(y):
            if P[index1] < p[index2, index1]:
                    P[index1] = p[index2, index1]
        totalPurity += m2[index1] * P[index1] / m
    #print('P = ', P)
    #print('totalPurity = ', totalPurity)
    
    return totalPurity


purity_GMMX = []
purity_GMMPCA = []
purity_GMMIsomap = []
purity_GMMLLE = []

purity_kmeansX = []
purity_kmeansPCA = []
purity_kmeansIsomap = []
purity_kmeansLLE = []

y = np.load('E:/Eduardo/Tese/Implementação/Indian Pines/Indian Pines Corrected Matrizes/Indian_pines_corrected_gt.npy')

for j in range (1,41):
    
    GMMXy = np.load('E:/Eduardo/Tese/Implementação/Indian Pines/Indian Pines Corrected Matrizes/GMM/Original/Not matched/Indian Pines_GMMX_execution'+str(j)+'.npy')
    GMMPCAy = np.load('E:/Eduardo/Tese/Implementação/Indian Pines/Indian Pines Corrected Matrizes/GMM/PCA/Not matched/Indian Pines_GMMPCA_execution'+str(j)+'.npy')
    GMMIsomapy = np.load('E:/Eduardo/Tese/Implementação/Indian Pines/Indian Pines Corrected Matrizes/GMM/Isomap/Not matched/Indian Pines_GMMIsomap_execution'+str(j)+'.npy')
    GMMLLEy = np.load('E:/Eduardo/Tese/Implementação/Indian Pines/Indian Pines Corrected Matrizes/GMM/LLE/Not matched/Indian Pines_GMMLLE_execution'+str(j)+'.npy')
    
    kmeansXy = np.load('E:/Eduardo/Tese/Implementação/Indian Pines/Indian Pines Corrected Matrizes/KMeans/Original/Not matched/Indian Pines_kmeansX_execution'+str(j)+'.npy')
    kmeansPCAy = np.load('E:/Eduardo/Tese/Implementação/Indian Pines/Indian Pines Corrected Matrizes/KMeans/PCA/Not matched/Indian Pines_kmeansPCA_execution'+str(j)+'.npy')
    kmeansIsomapy = np.load('E:/Eduardo/Tese/Implementação/Indian Pines/Indian Pines Corrected Matrizes/KMeans/Isomap/Not matched/Indian Pines_kmeansIsomap_execution'+str(j)+'.npy')
    kmeansLLEy = np.load('E:/Eduardo/Tese/Implementação/Indian Pines/Indian Pines Corrected Matrizes/KMeans/LLE/Not matched/Indian Pines_kmeansLLE_execution'+str(j)+'.npy')
    
    purity_GMMX.append(purity(y, GMMXy))
    purity_GMMPCA.append(purity(y, GMMPCAy))
    purity_GMMIsomap.append(purity(y, GMMIsomapy))
    purity_GMMLLE.append(purity(y, GMMLLEy))
    
    purity_kmeansX.append(purity(y, kmeansXy))
    purity_kmeansPCA.append(purity(y, kmeansPCAy))
    purity_kmeansIsomap.append(purity(y, kmeansIsomapy))
    purity_kmeansLLE.append(purity(y, kmeansLLEy))

purity_GMMX = np.array(purity_GMMX)
purity_GMMPCA = np.array(purity_GMMPCA)
purity_GMMIsomap = np.array(purity_GMMIsomap)
purity_GMMLLE = np.array(purity_GMMLLE)

purity_kmeansX = np.array(purity_kmeansX)
purity_kmeansPCA = np.array(purity_kmeansPCA)
purity_kmeansIsomap = np.array(purity_kmeansIsomap)
purity_kmeansLLE = np.array(purity_kmeansLLE)

result_directory = 'E:/Eduardo/Tese/Implementação/Indian Pines/Purity'
if not os.path.exists(result_directory):
    os.makedirs(result_directory)
    
np.save(result_directory+'/Indian Pines_purity_GMMX',purity_GMMX)
np.save(result_directory+'/Indian Pines_purity_GMMPCA',purity_GMMPCA)
np.save(result_directory+'/Indian Pines_purity_GMMIsomap',purity_GMMIsomap)
np.save(result_directory+'/Indian Pines_purity_GMMLLE',purity_GMMLLE)

np.save(result_directory+'/Indian Pines_purity_kmeansX',purity_kmeansX)
np.save(result_directory+'/Indian Pines_purity_kmeansPCA',purity_kmeansPCA)
np.save(result_directory+'/Indian Pines_purity_kmeansIsomap',purity_kmeansIsomap)
np.save(result_directory+'/Indian Pines_purity_kmeansLLE',purity_kmeansLLE)

purity_X = []
purity_PCA = []
purity_Isomap = []
purity_LLE = []

purity_X = np.array(purity_X)
purity_PCA = np.array(purity_PCA)
purity_Isomap = np.array(purity_Isomap)
purity_LLE = np.array(purity_LLE)

purity_X = np.concatenate((purity_kmeansX,purity_GMMX))
np.save(result_directory+'/Indian Pines_purity_X',purity_X)

purity_PCA = np.concatenate((purity_kmeansPCA, purity_GMMPCA))
np.save(result_directory+'/Indian Pines_purity_PCA',purity_PCA)

purity_Isomap = np.concatenate((purity_kmeansIsomap, purity_GMMIsomap))
np.save(result_directory+'/Indian Pines_purity_Isomap',purity_Isomap)

purity_LLE = np.concatenate((purity_kmeansLLE, purity_GMMLLE))
np.save(result_directory+'/Indian Pines_purity_LLE',purity_LLE)