#import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils import check_array
from sklearn.utils.graph import graph_shortest_path
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import KernelCenterer

class Isomap(BaseEstimator, TransformerMixin):

    def __init__(self, n_neighbors=5, n_components=2, eigen_solver='auto',
                 tol=0, max_iter=None, path_method='auto',
                 neighbors_algorithm='auto', n_jobs=1):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.path_method = path_method
        self.neighbors_algorithm = neighbors_algorithm
        self.n_jobs = n_jobs

    def _fit_transform(self, X):
        X = check_array(X)
        self.nbrs_ = NearestNeighbors(n_neighbors=self.n_neighbors,
                                      algorithm=self.neighbors_algorithm,
                                      n_jobs=self.n_jobs)
        self.nbrs_.fit(X)
        self.training_data_ = self.nbrs_._fit_X
        self.kernel_pca_ = KernelPCA(n_components=self.n_components,
                                     kernel="precomputed",
                                     eigen_solver=self.eigen_solver,
                                     tol=self.tol, max_iter=self.max_iter,
                                     n_jobs=self.n_jobs)
        kng = kneighbors_graph(self.nbrs_, self.n_neighbors,
                               mode='distance', n_jobs=self.n_jobs)
        self.dist_matrix_ = graph_shortest_path(kng,
                                                method=self.path_method,
                                                directed=False)
        G = self.dist_matrix_ ** 2
        G *= -0.5
        self.embedding_ = self.kernel_pca_.fit_transform(G)

    def reconstruction_error(self):
        G = -0.5 * self.dist_matrix_ ** 2
        G_center = KernelCenterer().fit_transform(G)
        evals = self.kernel_pca_.lambdas_
        return np.sqrt(np.sum(G_center ** 2) - np.sum(evals ** 2)) / G.shape[0]

    def fit(self, X, y=None):
        self._fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        self._fit_transform(X)
        return self.embedding_

    def transform(self, X):
        X = check_array(X)
        distances, indices = self.nbrs_.kneighbors(X, return_distance=True)
        G_X = np.zeros((X.shape[0], self.training_data_.shape[0]))
        for i in range(X.shape[0]):
            G_X[i] = np.min(self.dist_matrix_[indices[i]] +
                            distances[i][:, None], 0)
        G_X **= 2
        G_X *= -0.5
        return self.kernel_pca_.transform(G_X)

    def evals(self):
        evals = self.kernel_pca_.lambdas_
        return evals

AEVRbyBagging = []
##to be used only in bagging and background bagging cases
##each position 'i' stores the accumulated_explained_variance_ratio list for sampling 'i'lista = []

DimByBagging = []
##to be used only in bagging and background bagging cases
##each position 'i' stores the intrinsic dimensionality estimated for sampling 'i'

out_dim = 
iso = Isomap(n_components=out_dim,n_neighbors=2*out_dim)

for i in range(1,):
    X = np.load('C:/Users/Eduardo Kazuo Nakao//_reducedX_execution'+str(i)+'.npy')
    iso.fit(X)
    IsomapX = iso.transform(X)
    evals = iso.evals()
    accumulated_explained_variance_ratio = []
#    explained_variance_ratio = []
    for k in range(0,):
        accumulated_explained_variance_ratio.append(sum(evals[:k])/sum(evals))
#        explained_variance_ratio.append((evals[k])/sum(evals))
        if (sum(evals[:k])/sum(evals)) > 0.95:
#            print(k)
            AEVRbyBagging.append(accumulated_explained_variance_ratio)
            DimByBagging.append(k)
            break
    print(accumulated_explained_variance_ratio)


for i in range(0,):
    print(AEVRbyBagging[i])

#DimByBagging.sort()
#print(len(DimByBagging))
#print()
print(DimByBagging)


#    plt.figure(figsize=(100,5))
#    plt.plot(accumulated_explained_variance_ratio)
#    plt.xlim([0, 200])
#    plt.ylim([0, 1])
#    plt.yticks(np.arange(0, 1, 0.1))
#    plt.xticks(np.arange(1, 200, 1))
#    plt.show()
    
#    print("explained_variance_ratio")
#    print(explained_variance_ratio)
#    
#    plt.figure(figsize=(100,5))
#    plt.plot(explained_variance_ratio)
#    plt.xlim([0, 200])
#    plt.ylim([0, 1])
#    plt.yticks(np.arange(0, 1, 0.1))
#    plt.xticks(np.arange(1, 200, 1))
#    plt.show()