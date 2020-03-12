import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA
from scipy.sparse import linalg, eye
from scipy.sparse import csr_matrix
from scipy.linalg import solve
from pyamg import smoothed_aggregation_solver
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES

def barycenter_weights(X, Z, reg=1e-3): #get weights W that minimize the reconstruction error of a point from its kneighbors
    X = check_array(X, dtype=FLOAT_DTYPES)
    Z = check_array(Z, dtype=FLOAT_DTYPES, allow_nd=True)
    n_samples, n_neighbors = X.shape[0], Z.shape[1]
    B = np.empty((n_samples, n_neighbors), dtype=X.dtype)
    v = np.ones(n_neighbors, dtype=X.dtype)
    # this might raise a LinalgError if G is singular and has trace zero
    for i, A in enumerate(Z.transpose(0, 2, 1)):
        C = A.T - X[i]  # broadcasting
        G = np.dot(C, C.T)
        trace = np.trace(G)
        if trace > 0:
            R = reg * trace
        else:
            R = reg
        G.flat[::Z.shape[1] + 1] += R
        w = solve(G, v, sym_pos=True)
        B[i, :] = w / np.sum(w)
    return B

def barycenter_kneighbors_graph(X, n_neighbors, reg=1e-3, n_jobs=8): #get graph G from the data
    knn = NearestNeighbors(n_neighbors + 1, n_jobs=n_jobs).fit(X)
    X = knn._fit_X
    n_samples = X.shape[0]
    ind = knn.kneighbors(X, return_distance=False)[:, 1:]
    data = barycenter_weights(X, X[ind], reg=reg)
    indptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors)
    return csr_matrix((data.ravel(), ind.ravel(), indptr), shape=(n_samples, n_samples))

def locally_linear_embedding(X, n_neighbors, out_dim, tol=1e-6, max_iter=200):
    #W = neighbors.kneighbors_graph(X, n_neighbors=n_neighbors, mode='barycenter')
    W = barycenter_kneighbors_graph(X, n_neighbors=n_neighbors)
    # M = (I-W)' (I-W)
    A = eye(*W.shape, format=W.format) - W
    A = (A.T).dot(A).tocsr()
    # initial approximation to the eigenvectors
    initial_eigen_vectors = np.random.rand(W.shape[0], out_dim)
    ml = smoothed_aggregation_solver(A, symmetry='symmetric')
    prec = ml.aspreconditioner()
    # compute eigenvalues and eigenvectors with LOBPCG
    eigen_values, eigen_vectors = linalg.lobpcg(A, initial_eigen_vectors, M=prec, largest=False, tol=tol, maxiter=max_iter)
#    print('Eigenvalues')
#    print(eigen_values)
    index = np.argsort(eigen_values)
    return eigen_vectors[:, index], eigen_values, np.sum(eigen_values)


AEVRbyBagging = []
##to be used only in bagging and background bagging cases
##each position 'i' stores the accumulated_explained_variance_ratio list for sampling 'i'lista = []

DimByBagging = []
##to be used only in bagging and background bagging cases
##each position 'i' stores the intrinsic dimensionality estimated for sampling 'i'

out_dim = 
n_neighbors = 
#model = PCA()
for i in range(1,):
    X = np.load('C:/Users/Eduardo Kazuo Nakao//_reducedX_execution'+str(i)+'.npy')
    #model.fit(X)
    #PCAX = model.transform(X)
    X_r, evals, cost = locally_linear_embedding(X, n_neighbors, out_dim)
    accumulated_explained_variance_ratio = []
#    explained_variance_ratio = []
    for k in range(0,):
        accumulated_explained_variance_ratio.append(sum(evals[:k])/sum(evals))
#        explained_variance_ratio.append((evals[k])/sum(evals))
        if (sum(evals[:k])/sum(evals)) > 0.01:
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