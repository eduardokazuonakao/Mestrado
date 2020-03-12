import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding

model = PCA(n_components=)
iso = Isomap(n_neighbors=, n_components=)
clf = LocallyLinearEmbedding(n_neighbors=, n_components=, method='standard')

for i in range(1,41):
    X = np.load('C:/Users/Eduardo Kazuo Nakao/Botswana background bagging 5%/Botswana_reducedX_execution'+str(i)+'.npy')
    model.fit(X)
    PCAX = model.transform(X)
    np.save('C:/Users/Eduardo Kazuo Nakao/Botswana background bagging 5%/PCA/Botswana_PCA_execution'+str(i),PCAX)
    
print("fim PCA")

for i in range(1,41):
    X = np.load('C:/Users/Eduardo Kazuo Nakao/Botswana background bagging 5%/Botswana_reducedX_execution'+str(i)+'.npy')
    iso.fit(X)
    IsomapX = iso.transform(X)
    np.save('C:/Users/Eduardo Kazuo Nakao/Botswana background bagging 5%/Isomap/Botswana_Isomap_execution'+str(i),IsomapX)
    
print("fim Isomap")
    
for i in range(1,41):
    X = np.load('C:/Users/Eduardo Kazuo Nakao/Botswana background bagging 5%/Botswana_reducedX_execution'+str(i)+'.npy')
    LLEX = clf.fit_transform(X)
    np.save('C:/Users/Eduardo Kazuo Nakao/Botswana background bagging 5%/LLE/Botswana_LLE_execution'+str(i),LLEX)

"""Execution times"""
##    print("calculating PCA...")
##    t0 = time()
#    model = PCA(n_components=)
#    model.fit(X)
#    PCAX = model.transform(X)
##    t1 = time()
##    print("PCA: %.2g sec" % (t1 - t0))
    
#    print("calculating Isomap...")
#    t0 = time()
#    iso = Isomap(n_components=4)
#    iso.fit(X)
#    IsomapX = iso.transform(X)
#    t1 = time()
#    print("Isomap: %.2g sec" % (t1 - t0))
    
#    print("calculating LLE...")
#    t0 = time()
#    clf = LocallyLinearEmbedding(n_neighbors=400, n_components=8, method='standard')
#    LLEX = clf.fit_transform(X)
#    t1 = time()
#    print("LLE: %.2g sec" % (t1 - t0))