import numpy as np
from sklearn.metrics import cohen_kappa_score
import os

kappa_GMMX = []
kappa_GMMPCA = []
kappa_GMMIsomap = []
kappa_GMMLLE = []

kappa_kmeansX = []
kappa_kmeansPCA = []
kappa_kmeansIsomap = []
kappa_kmeansLLE = []

y = np.load('E:/Eduardo/Tese/Implementação/Indian Pines/Indian Pines Corrected Matrizes/Indian_pines_corrected_gt.npy')
   
for j in range (1, 41):
    
    GMMXy = np.load('E:/Eduardo/Tese/Implementação/Indian Pines/Indian Pines Corrected Matrizes/GMM/Original/Matched/Indian Pines_GMMX_execution'+str(j)+'_matched.npy')
    GMMPCAy = np.load('E:/Eduardo/Tese/Implementação/Indian Pines/Indian Pines Corrected Matrizes/GMM/PCA/Matched/Indian Pines_GMMPCA_execution'+str(j)+'_matched.npy')
    GMMIsomapy = np.load('E:/Eduardo/Tese/Implementação/Indian Pines/Indian Pines Corrected Matrizes/GMM/Isomap/Matched/Indian Pines_GMMIsomap_execution'+str(j)+'_matched.npy')
    GMMLLEy = np.load('E:/Eduardo/Tese/Implementação/Indian Pines/Indian Pines Corrected Matrizes/GMM/LLE/Matched/Indian Pines_GMMLLE_execution'+str(j)+'_matched.npy')
    
    kmeansXy = np.load('E:/Eduardo/Tese/Implementação/Indian Pines/Indian Pines Corrected Matrizes/KMeans/Original/Matched/Indian Pines_kmeansX_execution'+str(j)+'_matched.npy')
    kmeansPCAy = np.load('E:/Eduardo/Tese/Implementação/Indian Pines/Indian Pines Corrected Matrizes/KMeans/PCA/Matched/Indian Pines_kmeansPCA_execution'+str(j)+'_matched.npy')
    kmeansIsomapy = np.load('E:/Eduardo/Tese/Implementação/Indian Pines/Indian Pines Corrected Matrizes/KMeans/Isomap/Matched/Indian Pines_kmeansIsomap_execution'+str(j)+'_matched.npy')
    kmeansLLEy = np.load('E:/Eduardo/Tese/Implementação/Indian Pines/Indian Pines Corrected Matrizes/KMeans/LLE/Matched/Indian Pines_kmeansLLE_execution'+str(j)+'_matched.npy')
    
    kappa_GMMX.append(cohen_kappa_score(y, GMMXy))
    kappa_GMMPCA.append(cohen_kappa_score(y, GMMPCAy))
    kappa_GMMIsomap.append(cohen_kappa_score(y, GMMIsomapy))
    kappa_GMMLLE.append(cohen_kappa_score(y, GMMLLEy))
    
    kappa_kmeansX.append(cohen_kappa_score(y, kmeansXy))
    kappa_kmeansPCA.append(cohen_kappa_score(y, kmeansPCAy))
    kappa_kmeansIsomap.append(cohen_kappa_score(y, kmeansIsomapy))
    kappa_kmeansLLE.append(cohen_kappa_score(y, kmeansLLEy))

kappa_GMMX = np.array(kappa_GMMX)
kappa_GMMPCA = np.array(kappa_GMMPCA)
kappa_GMMIsomap = np.array(kappa_GMMIsomap)
kappa_GMMLLE = np.array(kappa_GMMLLE)

kappa_kmeansX = np.array(kappa_kmeansX)
kappa_kmeansPCA = np.array(kappa_kmeansPCA)
kappa_kmeansIsomap = np.array(kappa_kmeansIsomap)
kappa_kmeansLLE = np.array(kappa_kmeansLLE)

result_directory = 'E:/Eduardo/Tese/Implementação/Indian Pines/Kappa/Matched'
if not os.path.exists(result_directory):
    os.makedirs(result_directory)
    
np.save(result_directory+'/Indian Pines_kappa_GMMX_matched',kappa_GMMX)
np.save(result_directory+'/Indian Pines_kappa_GMMPCA_matched',kappa_GMMPCA)
np.save(result_directory+'/Indian Pines_kappa_GMMIsomap_matched',kappa_GMMIsomap)
np.save(result_directory+'/Indian Pines_kappa_GMMLLE_matched',kappa_GMMLLE)

np.save(result_directory+'/Indian Pines_kappa_kmeansX_matched',kappa_kmeansX)
np.save(result_directory+'/Indian Pines_kappa_kmeansPCA_matched',kappa_kmeansPCA)
np.save(result_directory+'/Indian Pines_kappa_kmeansIsomap_matched',kappa_kmeansIsomap)
np.save(result_directory+'/Indian Pines_kappa_kmeansLLE_matched',kappa_kmeansLLE)

kappa_X = []
kappa_PCA = []
kappa_Isomap = []
kappa_LLE = []

kappa_X = np.array(kappa_X)
kappa_PCA = np.array(kappa_PCA)
kappa_Isomap = np.array(kappa_Isomap)
kappa_LLE = np.array(kappa_LLE)

kappa_X = np.concatenate((kappa_kmeansX,kappa_GMMX))
np.save(result_directory+'/Indian Pines_kappa_X_matched',kappa_X)

kappa_PCA = np.concatenate((kappa_kmeansPCA, kappa_GMMPCA))
np.save(result_directory+'/Indian Pines_kappa_PCA_matched',kappa_PCA)

kappa_Isomap = np.concatenate((kappa_kmeansIsomap, kappa_GMMIsomap))
np.save(result_directory+'/Indian Pines_kappa_Isomap_matched',kappa_Isomap)

kappa_LLE = np.concatenate((kappa_kmeansLLE, kappa_GMMLLE))
np.save(result_directory+'/Indian Pines_kappa_LLE_matched',kappa_LLE)