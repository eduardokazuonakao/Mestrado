from sklearn.metrics import silhouette_score
import numpy as np
import os

silhouette_GMMX = []
silhouette_GMMPCA = []
silhouette_GMMIsomap = []
silhouette_GMMLLE = []

silhouette_kmeansX = []
silhouette_kmeansPCA = []
silhouette_kmeansIsomap = []
silhouette_kmeansLLE = []

silhouette_X = []
silhouette_PCA = []
silhouette_Isomap = []
silhouette_LLE = []

silhouette_X = np.array(silhouette_X)
silhouette_PCA = np.array(silhouette_PCA)
silhouette_Isomap = np.array(silhouette_Isomap)
silhouette_LLE = np.array(silhouette_LLE)

result_directory = 'E:/Eduardo/Tese/Implementação/Botswana/Silhouette'
if not os.path.exists(result_directory):
    os.makedirs(result_directory)

for i in range (1,41):

    X = np.load('E:/Eduardo/Tese/Implementação/Botswana/Botswana Matrizes/Background bagging/Botswana_reducedX_execution'+str(i)+'.npy')
    PCA = np.load('E:/Eduardo/Tese/Implementação/Botswana/Botswana Matrizes/PCA/Botswana_PCA_execution'+str(i)+'.npy')
    Isomap = np.load('E:/Eduardo/Tese/Implementação/Botswana/Botswana Matrizes/Isomap/Botswana_Isomap_execution'+str(i)+'.npy')
    LLE = np.load('E:/Eduardo/Tese/Implementação/Botswana/Botswana Matrizes/LLE/Botswana_LLE_execution'+str(i)+'.npy')

    for j in range (1,41):

        GMMXy = np.load('E:/Eduardo/Tese/Implementação/Botswana/Botswana Matrizes/GMM/Original/Reduction'+str(i)+'/Not matched/Botswana_GMMX_reduction'+str(i)+'_clustering'+str(j)+'.npy')
        silhouette_GMMX.append(silhouette_score(X, GMMXy))
        silhouette_GMMX_array = np.array(silhouette_GMMX)
        np.save(result_directory+'/Botswana_silhouette_GMMX',silhouette_GMMX_array)

        GMMPCAy = np.load('E:/Eduardo/Tese/Implementação/Botswana/Botswana Matrizes/GMM/PCA/Reduction'+str(i)+'/Not matched/Botswana_GMMPCA_reduction'+str(i)+'_clustering'+str(j)+'.npy')
        silhouette_GMMPCA.append(silhouette_score(PCA, GMMPCAy))
        silhouette_GMMPCA_array = np.array(silhouette_GMMPCA)
        np.save(result_directory+'/Botswana_silhouette_GMMPCA',silhouette_GMMPCA_array)
        
        GMMIsomapy = np.load('E:/Eduardo/Tese/Implementação/Botswana/Botswana Matrizes/GMM/Isomap/Reduction'+str(i)+'/Not matched/Botswana_GMMIsomap_reduction'+str(i)+'_clustering'+str(j)+'.npy')
        silhouette_GMMIsomap.append(silhouette_score(Isomap, GMMIsomapy))
        silhouette_GMMIsomap_array = np.array(silhouette_GMMIsomap)
        np.save(result_directory+'/Botswana_silhouette_GMMIsomap',silhouette_GMMIsomap_array)

        GMMLLEy = np.load('E:/Eduardo/Tese/Implementação/Botswana/Botswana Matrizes/GMM/LLE/Reduction'+str(i)+'/Not matched/Botswana_GMMLLE_reduction'+str(i)+'_clustering'+str(j)+'.npy')
        silhouette_GMMLLE.append(silhouette_score(LLE, GMMLLEy))
        silhouette_GMMLLE_array = np.array(silhouette_GMMLLE)
        np.save(result_directory+'/Botswana_silhouette_GMMLLE',silhouette_GMMLLE_array)

        kmeansXy = np.load('E:/Eduardo/Tese/Implementação/Botswana/Botswana Matrizes/KMeans/Original/Reduction'+str(i)+'/Not matched/Botswana_kmeansX_reduction'+str(i)+'_clustering'+str(j)+'.npy')
        silhouette_kmeansX.append(silhouette_score(X, kmeansXy))
        silhouette_kmeansX_array = np.array(silhouette_kmeansX)
        np.save(result_directory+'/Botswana_silhouette_kmeansX',silhouette_kmeansX_array)

        kmeansPCAy = np.load('E:/Eduardo/Tese/Implementação/Botswana/Botswana Matrizes/KMeans/PCA/Reduction'+str(i)+'/Not matched/Botswana_kmeansPCA_reduction'+str(i)+'_clustering'+str(j)+'.npy')
        silhouette_kmeansPCA.append(silhouette_score(PCA, kmeansPCAy))
        silhouette_kmeansPCA_array = np.array(silhouette_kmeansPCA)
        np.save(result_directory+'/Botswana_silhouette_kmeansPCA',silhouette_kmeansPCA_array)

        kmeansIsomapy = np.load('E:/Eduardo/Tese/Implementação/Botswana/Botswana Matrizes/KMeans/Isomap/Reduction'+str(i)+'/Not matched/Botswana_kmeansIsomap_reduction'+str(i)+'_clustering'+str(j)+'.npy')
        silhouette_kmeansIsomap.append(silhouette_score(Isomap, kmeansIsomapy))
        silhouette_kmeansIsomap_array = np.array(silhouette_kmeansIsomap)
        np.save(result_directory+'/Botswana_silhouette_kmeansIsomap',silhouette_kmeansIsomap_array)

        kmeansLLEy = np.load('E:/Eduardo/Tese/Implementação/Botswana/Botswana Matrizes/KMeans/LLE/Reduction'+str(i)+'/Not matched/Botswana_kmeansLLE_reduction'+str(i)+'_clustering'+str(j)+'.npy')
        silhouette_kmeansLLE.append(silhouette_score(LLE, kmeansLLEy))
        silhouette_kmeansLLE_array = np.array(silhouette_kmeansLLE)
        np.save(result_directory+'/Botswana_silhouette_kmeansLLE',silhouette_kmeansLLE_array)


        silhouette_X = np.concatenate((silhouette_kmeansX_array,silhouette_GMMX_array))
        np.save(result_directory+'/Botswana_silhouette_X',silhouette_X)

        silhouette_PCA = np.concatenate((silhouette_kmeansPCA_array, silhouette_GMMPCA_array))
        np.save(result_directory+'/Botswana_silhouette_PCA',silhouette_PCA)

        silhouette_Isomap = np.concatenate((silhouette_kmeansIsomap_array, silhouette_GMMIsomap_array))
        np.save(result_directory+'/Botswana_silhouette_Isomap',silhouette_Isomap)

        silhouette_LLE = np.concatenate((silhouette_kmeansLLE_array, silhouette_GMMLLE_array))
        np.save(result_directory+'/Botswana_silhouette_LLE',silhouette_LLE)