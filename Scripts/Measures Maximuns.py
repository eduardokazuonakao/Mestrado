import numpy as np

Rand_NDR = np.load('E:/Eduardo/Tese/Implementação/Salinas/Dimensionalidade aumentada/Rand//Salinas_rand_X.npy')
Rand_PCA = np.load('E:/Eduardo/Tese/Implementação/Salinas/Dimensionalidade aumentada/Rand//Salinas_rand_PCA.npy')
Rand_Isomap = np.load('E:/Eduardo/Tese/Implementação/Salinas/Dimensionalidade aumentada/Rand//Salinas_rand_Isomap.npy')
Rand_LLE = np.load('E:/Eduardo/Tese/Implementação/Salinas/Dimensionalidade aumentada/Rand//Salinas_rand_LLE.npy')

print(np.amax(Rand_NDR))
print(np.amax(Rand_PCA))
print(np.amax(Rand_Isomap))
print(np.amax(Rand_LLE))