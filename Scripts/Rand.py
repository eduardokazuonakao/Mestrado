import numpy as np
from sklearn.metrics import adjusted_rand_score
import os

rand_GMMX = []
rand_GMMPCA = []
rand_GMMIsomap = []
rand_GMMLLE = []

rand_kmeansX = []
rand_kmeansPCA = []
rand_kmeansIsomap = []
rand_kmeansLLE = []

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

        rand_GMMX.append(adjusted_rand_score(y, GMMXy))
        rand_GMMPCA.append(adjusted_rand_score(y, GMMPCAy))
        rand_GMMIsomap.append(adjusted_rand_score(y, GMMIsomapy))
        rand_GMMLLE.append(adjusted_rand_score(y, GMMLLEy))

        rand_kmeansX.append(adjusted_rand_score(y, kmeansXy))
        rand_kmeansPCA.append(adjusted_rand_score(y, kmeansPCAy))
        rand_kmeansIsomap.append(adjusted_rand_score(y, kmeansIsomapy))
        rand_kmeansLLE.append(adjusted_rand_score(y, kmeansLLEy))

rand_GMMX = np.array(rand_GMMX)
rand_GMMPCA = np.array(rand_GMMPCA)
rand_GMMIsomap = np.array(rand_GMMIsomap)
rand_GMMLLE = np.array(rand_GMMLLE)

rand_kmeansX = np.array(rand_kmeansX)
rand_kmeansPCA = np.array(rand_kmeansPCA)
rand_kmeansIsomap = np.array(rand_kmeansIsomap)
rand_kmeansLLE = np.array(rand_kmeansLLE)

result_directory = 'E:/Eduardo/Tese/Implementação/Botswana/Rand'
if not os.path.exists(result_directory):
    os.makedirs(result_directory)

np.save(result_directory+'/Botswana_rand_GMMX',rand_GMMX)
np.save(result_directory+'/Botswana_rand_GMMPCA',rand_GMMPCA)
np.save(result_directory+'/Botswana_rand_GMMIsomap',rand_GMMIsomap)
np.save(result_directory+'/Botswana_rand_GMMLLE',rand_GMMLLE)

np.save(result_directory+'/Botswana_rand_kmeansX',rand_kmeansX)
np.save(result_directory+'/Botswana_rand_kmeansPCA',rand_kmeansPCA)
np.save(result_directory+'/Botswana_rand_kmeansIsomap',rand_kmeansIsomap)
np.save(result_directory+'/Botswana_rand_kmeansLLE',rand_kmeansLLE)

rand_X = []
rand_PCA = []
rand_Isomap = []
rand_LLE = []

rand_X = np.array(rand_X)
rand_PCA = np.array(rand_PCA)
rand_Isomap = np.array(rand_Isomap)
rand_LLE = np.array(rand_LLE)

rand_X = np.concatenate((rand_kmeansX,rand_GMMX))
np.save(result_directory+'/Botswana_rand_X',rand_X)

rand_PCA = np.concatenate((rand_kmeansPCA, rand_GMMPCA))
np.save(result_directory+'/Botswana_rand_PCA',rand_PCA)

rand_Isomap = np.concatenate((rand_kmeansIsomap, rand_GMMIsomap))
np.save(result_directory+'/Botswana_rand_Isomap',rand_Isomap)

rand_LLE = np.concatenate((rand_kmeansLLE, rand_GMMLLE))
np.save(result_directory+'/Botswana_rand_LLE',rand_LLE)