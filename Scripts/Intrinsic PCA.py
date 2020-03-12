import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA

AEVRbyBagging = []
##to be used only in bagging and background bagging cases
##each position 'i' stores the accumulated_explained_variance_ratio list for sampling 'i'

DimByBagging = []
##to be used only in bagging and background bagging cases
##each position 'i' stores the intrinsic dimensionality estimated for sampling 'i'

"""using covariance matrix"""  
for i in range (1,):
    X = np.load('C:/Users/Eduardo Kazuo Nakao//_reducedX_execution'+str(i)+'.npy')
    cov_mat = np.cov(X.T)
    print(cov_mat.shape)
    evals, eigen_vecs = np.linalg.eig(cov_mat)
#    explained_variance_ratio = []
    accumulated_explained_variance_ratio = []
    for k in range(0,):
#        explained_variance_ratio.append((desc_evals[k])/sum(desc_evals))
        accumulated_explained_variance_ratio.append(sum(evals[:k])/sum(evals))
        if (sum(evals[:k])/sum(evals)) > 0.95:
#            print(k)
            AEVRbyBagging.append(accumulated_explained_variance_ratio)
            DimByBagging.append(k)
            break
#    print(accumulated_explained_variance_ratio)


for i in range(0,):
    print(AEVRbyBagging[i])

#DimByBagging.sort()
#print(len(DimByBagging))
#print()
print(DimByBagging)


   ##ploting the ith eigenvector in x axis and its explained variance ratio in the y axis
#    plt.figure(figsize=(100,5))
#    plt.plot(explained_variance_ratio)
#    plt.xlim([0, 20])
#    plt.ylim([0, 1])
#    plt.yticks(np.arange(0, 1, 0.1))
#    plt.xticks(np.arange(0, 200, 1))
#    plt.show()



"""using PCA"""
#model = PCA() #no parameter will project in original dimension
#model.fit(preReducedX)
#PCAX = model.transform(preReducedX)

##explained_variance_ratio_ : array, shape (n_components,)
    ##Percentage of variance explained by each of the selected components.
    ##If n_components is not set then all components are stored and the sum of explained variances is equal to 1.0.

#plt.figure(figsize=(100,5))
#plt.plot(model.explained_variance_ratio_)
#plt.xlim([0, 200])
#plt.ylim([0, 1])
#plt.yticks(np.arange(0, 1, 0.1))
#plt.xticks(np.arange(0, 200, 1))
#plt.show()