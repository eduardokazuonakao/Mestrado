import numpy as np
from sklearn.metrics import jaccard_score
import os

jaccard_GMMX = []
jaccard_GMMPCA = []
jaccard_GMMIsomap = []
jaccard_GMMLLE = []

jaccard_kmeansX = []
jaccard_kmeansPCA = []
jaccard_kmeansIsomap = []
jaccard_kmeansLLE = []

y = np.load('E:/Eduardo/Tese/Implementação/Indian Pines/Indian Pines Matrizes/Indian Pines_corrected_gt.npy')

for j in range (1,41):

    GMMXy = np.load('E:/Eduardo/Tese/Implementação/Indian Pines/Indian Pines Matrizes/GMM/Original/Not matched/Indian Pines_GMMX_execution'+str(j)+'.npy')
    GMMPCAy = np.load('E:/Eduardo/Tese/Implementação/Indian Pines/Indian Pines Matrizes/GMM/PCA/Not matched/Indian Pines_GMMPCA_execution'+str(j)+'.npy')
    GMMIsomapy = np.load('E:/Eduardo/Tese/Implementação/Indian Pines/Indian Pines Matrizes/GMM/Isomap/Not matched/Indian Pines_GMMIsomap_execution'+str(j)+'.npy')
    GMMLLEy = np.load('E:/Eduardo/Tese/Implementação/Indian Pines/Indian Pines Matrizes/GMM/LLE/Not matched/Indian Pines_GMMLLE_execution'+str(j)+'.npy')

    kmeansXy = np.load('E:/Eduardo/Tese/Implementação/Indian Pines/Indian Pines Matrizes/KMeans/Original/Not matched/Indian Pines_kmeansX_execution'+str(j)+'.npy')
    kmeansPCAy = np.load('E:/Eduardo/Tese/Implementação/Indian Pines/Indian Pines Matrizes/KMeans/PCA/Not matched/Indian Pines_kmeansPCA_execution'+str(j)+'.npy')
    kmeansIsomapy = np.load('E:/Eduardo/Tese/Implementação/Indian Pines/Indian Pines Matrizes/KMeans/Isomap/Not matched/Indian Pines_kmeansIsomap_execution'+str(j)+'.npy')
    kmeansLLEy = np.load('E:/Eduardo/Tese/Implementação/Indian Pines/Indian Pines Matrizes/KMeans/LLE/Not matched/Indian Pines_kmeansLLE_execution'+str(j)+'.npy')

    jaccard_GMMX.append(jaccard_score(y, GMMXy, average='micro'))
    jaccard_GMMPCA.append(jaccard_score(y, GMMPCAy, average='micro'))
    jaccard_GMMIsomap.append(jaccard_score(y, GMMIsomapy, average='micro'))
    jaccard_GMMLLE.append(jaccard_score(y, GMMLLEy, average='micro'))

    jaccard_kmeansX.append(jaccard_score(y, kmeansXy, average='micro'))
    jaccard_kmeansPCA.append(jaccard_score(y, kmeansPCAy, average='micro'))
    jaccard_kmeansIsomap.append(jaccard_score(y, kmeansIsomapy, average='micro'))
    jaccard_kmeansLLE.append(jaccard_score(y, kmeansLLEy, average='micro'))

jaccard_GMMX = np.array(jaccard_GMMX)
jaccard_GMMPCA = np.array(jaccard_GMMPCA)
jaccard_GMMIsomap = np.array(jaccard_GMMIsomap)
jaccard_GMMLLE = np.array(jaccard_GMMLLE)

jaccard_kmeansX = np.array(jaccard_kmeansX)
jaccard_kmeansPCA = np.array(jaccard_kmeansPCA)
jaccard_kmeansIsomap = np.array(jaccard_kmeansIsomap)
jaccard_kmeansLLE = np.array(jaccard_kmeansLLE)

result_directory = 'E:/Eduardo/Tese/Implementação/Indian Pines/Jaccard'
if not os.path.exists(result_directory):
    os.makedirs(result_directory)

np.save(result_directory+'/Indian Pines_jaccard_GMMX',jaccard_GMMX)
np.save(result_directory+'/Indian Pines_jaccard_GMMPCA',jaccard_GMMPCA)
np.save(result_directory+'/Indian Pines_jaccard_GMMIsomap',jaccard_GMMIsomap)
np.save(result_directory+'/Indian Pines_jaccard_GMMLLE',jaccard_GMMLLE)

np.save(result_directory+'/Indian Pines_jaccard_kmeansX',jaccard_kmeansX)
np.save(result_directory+'/Indian Pines_jaccard_kmeansPCA',jaccard_kmeansPCA)
np.save(result_directory+'/Indian Pines_jaccard_kmeansIsomap',jaccard_kmeansIsomap)
np.save(result_directory+'/Indian Pines_jaccard_kmeansLLE',jaccard_kmeansLLE)

jaccard_X = []
jaccard_PCA = []
jaccard_Isomap = []
jaccard_LLE = []

jaccard_X = np.array(jaccard_X)
jaccard_PCA = np.array(jaccard_PCA)
jaccard_Isomap = np.array(jaccard_Isomap)
jaccard_LLE = np.array(jaccard_LLE)

jaccard_X = np.concatenate((jaccard_kmeansX,jaccard_GMMX))
np.save(result_directory+'/Indian Pines_jaccard_X',jaccard_X)

jaccard_PCA = np.concatenate((jaccard_kmeansPCA, jaccard_GMMPCA))
np.save(result_directory+'/Indian Pines_jaccard_PCA',jaccard_PCA)

jaccard_Isomap = np.concatenate((jaccard_kmeansIsomap, jaccard_GMMIsomap))
np.save(result_directory+'/Indian Pines_jaccard_Isomap',jaccard_Isomap)

jaccard_LLE = np.concatenate((jaccard_kmeansLLE, jaccard_GMMLLE))
np.save(result_directory+'/Indian Pines_jaccard_LLE',jaccard_LLE)