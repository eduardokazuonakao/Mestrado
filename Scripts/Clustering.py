import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import scipy.optimize
import os


"""sets to 0 the coordinates of points that belongs to class 0 in parameter matrix X"""
def normalizeBackground(parameterX,y):
    for i in range(0,y.size):
        if y[i] == 0:
            parameterX[i] = 0


"""plots the color map of parameter classes matrix"""
#def colorMap(y):
##    transforms clustering classes vector in matrix;
#    yMatrix = y.reshape(512,217)
#    print(yMatrix)
#    print(yMatrix.shape)
##    plot matrix as image; in this matrix the first sample corresponds to the first pixel of the image,
##       the second to the second, and so forth...
#    plt.imshow(yMatrix, cmap=plt.cm.get_cmap('Spectral'))
#    plt.colorbar(ticks = range(0,len(y)))


"""function that receives gt classes and clustering classes and do the best matching between its labels"""
def labelMatching(y,clustering):
    nC = len(set(y))
    M = np.zeros((nC, nC))
    for i in range(nC):
        #for each iteration i, indexes_A stores the indexes of the elements of clustering that are equals to i
        indexes_A = np.where(clustering==i)
        nI_A = len(indexes_A[0])
        for j in range(nC):
            indexes_B = np.where(y==j)
            nI_B = len(indexes_B[0])
            set_A = set(list(indexes_A[0]))
            set_B = set(list(indexes_B[0]))
            intersectAB = len(set_A.intersection(set_B))
            M[j,i] = (nI_A - intersectAB) + (nI_B - intersectAB)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(M)
#    print('Mapping')
#    print(row_ind)
#    print(col_ind)
    col_ind = list(col_ind)
    for i in range(0,clustering.size):
        index = col_ind.index(clustering[i])
        clustering[i] = row_ind[index]
    return clustering


model = GaussianMixture(n_components=15,covariance_type='full',reg_covar=1e-05)
kmeans = KMeans(n_clusters=15)

for i in range (1,41):

    y = np.load('E:/Eduardo/Botswana/Botswana_reducedY.npy')
    X = np.load('E:/Eduardo/Botswana/Background bagging/Botswana_reducedX_execution'+str(i)+'.npy')
    PCAX = np.load('E:/Eduardo/Botswana/PCA/Botswana_PCA_execution'+str(i)+'.npy')
    IsomapX = np.load('E:/Eduardo/Botswana/Isomap/Botswana_Isomap_execution'+str(i)+'.npy')
    LLEX = np.load('E:/Eduardo/Botswana/LLE/Botswana_LLE_execution'+str(i)+'.npy')
    
    normalizeBackground(X,y)
    normalizeBackground(PCAX,y)
    normalizeBackground(IsomapX,y)
    normalizeBackground(LLEX,y)
    
    for j in range (1,41):
    
        model.fit(X)
        GMMXy = model.predict(X)
        result_directory = 'E:/Eduardo/Botswana/GMM/Original/Reduction'+str(i)+'/Not matched'
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        np.save(result_directory+'/Botswana_GMMX_reduction'+str(i)+'_clustering'+str(j),GMMXy)
        GMMXy = labelMatching(y,GMMXy)
        result_directory = 'E:/Eduardo/Botswana/GMM/Original/Reduction'+str(i)+'/Matched'
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        np.save(result_directory+'/Botswana_GMMX_reduction'+str(i)+'_clustering'+str(j)+'_matched',GMMXy)
    
        model.fit(PCAX)
        GMMPCAy = model.predict(PCAX)
        result_directory = 'E:/Eduardo/Botswana/GMM/PCA/Reduction'+str(i)+'/Not matched'
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        np.save(result_directory+'/Botswana_GMMPCA_reduction'+str(i)+'_clustering'+str(j),GMMPCAy)
        GMMPCAy = labelMatching(y,GMMPCAy)
        result_directory = 'E:/Eduardo/Botswana/GMM/PCA/Reduction'+str(i)+'/Matched'
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        np.save(result_directory+'/Botswana_GMMPCA_reduction'+str(i)+'_clustering'+str(j)+'_matched',GMMPCAy)
    
        model.fit(IsomapX)
        GMMIsomapy = model.predict(IsomapX)
        result_directory = 'E:/Eduardo/Botswana/GMM/Isomap/Reduction'+str(i)+'/Not matched'
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        np.save(result_directory+'/Botswana_GMMIsomap_reduction'+str(i)+'_clustering'+str(j),GMMIsomapy)
        GMMIsomapy = labelMatching(y,GMMIsomapy)
        result_directory = 'E:/Eduardo/Botswana/GMM/Isomap/Reduction'+str(i)+'/Matched'
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        np.save(result_directory+'/Botswana_GMMIsomap_reduction'+str(i)+'_clustering'+str(j)+'_matched',GMMIsomapy)
    
        model.fit(LLEX)
        GMMLLEy = model.predict(LLEX)
        result_directory = 'E:/Eduardo/Botswana/GMM/LLE/Reduction'+str(i)+'/Not matched'
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        np.save(result_directory+'/Botswana_GMMLLE_reduction'+str(i)+'_clustering'+str(j),GMMLLEy)
        GMMLLEy = labelMatching(y,GMMLLEy)
        result_directory = 'E:/Eduardo/Botswana/GMM/LLE/Reduction'+str(i)+'/Matched'
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        np.save(result_directory+'/Botswana_GMMLLE_reduction'+str(i)+'_clustering'+str(j)+'_matched',GMMLLEy)
    
        kmeans.fit(X)
        kmeansXy = kmeans.predict(X)
        result_directory = 'E:/Eduardo/Botswana/KMeans/Original/Reduction'+str(i)+'/Not matched'
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        np.save(result_directory+'/Botswana_kmeansX_reduction'+str(i)+'_clustering'+str(j),kmeansXy)
        kmeansXy = labelMatching(y,kmeansXy)
        result_directory = 'E:/Eduardo/Botswana/KMeans/Original/Reduction'+str(i)+'/Matched'
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        np.save(result_directory+'/Botswana_kmeansX_reduction'+str(i)+'_clustering'+str(j)+'_matched',kmeansXy)
    
        kmeans.fit(PCAX)
        kmeansPCAy = kmeans.predict(PCAX)
        result_directory = 'E:/Eduardo/Botswana/KMeans/PCA/Reduction'+str(i)+'/Not matched'
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        np.save(result_directory+'/Botswana_kmeansPCA_reduction'+str(i)+'_clustering'+str(j),kmeansPCAy)
        kmeansPCAy = labelMatching(y,kmeansPCAy)
        result_directory = 'E:/Eduardo/Botswana/KMeans/PCA/Reduction'+str(i)+'/Matched'
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        np.save(result_directory+'/Botswana_kmeansPCA_reduction'+str(i)+'_clustering'+str(j)+'_matched',kmeansPCAy)
    
        kmeans.fit(IsomapX)
        kmeansIsomapy = kmeans.predict(IsomapX)
        result_directory = 'E:/Eduardo/Botswana/KMeans/Isomap/Reduction'+str(i)+'/Not matched'
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        np.save(result_directory+'/Botswana_kmeansIsomap_reduction'+str(i)+'_clustering'+str(j),kmeansIsomapy)
        kmeansIsomapy = labelMatching(y,kmeansIsomapy)
        result_directory = 'E:/Eduardo/Botswana/KMeans/Isomap/Reduction'+str(i)+'/Matched'
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        np.save(result_directory+'/Botswana_kmeansIsomap_reduction'+str(i)+'_clustering'+str(j)+'_matched',kmeansIsomapy)
    
        kmeans.fit(LLEX)
        kmeansLLEy = kmeans.predict(LLEX)
        result_directory = 'E:/Eduardo/Botswana/KMeans/LLE/Reduction'+str(i)+'/Not matched'
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        np.save(result_directory+'/Botswana_kmeansLLE_reduction'+str(i)+'_clustering'+str(j),kmeansLLEy)
        kmeansLLEy = labelMatching(y,kmeansLLEy)
        result_directory = 'E:/Eduardo/Botswana/KMeans/LLE/Reduction'+str(i)+'/Matched'
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        np.save(result_directory+'/Botswana_kmeansLLE_reduction'+str(i)+'_clustering'+str(j)+'_matched',kmeansLLEy)