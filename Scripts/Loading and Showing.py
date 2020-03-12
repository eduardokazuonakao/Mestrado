import scipy.io
import matplotlib.pyplot as plt
import numpy as np

"""loading, formating and showing data matrix X"""
data = scipy.io.loadmat('C:/Users/Eduardo Kazuo Nakao/Tese/Implementação/Scenes/KSC.mat')
print(data)
dataMatrix = data['KSC']
print(dataMatrix) #print the nparray format
print(dataMatrix.shape)
#below is a matplotlib command to show the image. last index can vary until number of bands -1
#    first or second index can be fixed with third index ':' to see full spectrum of fixed pixel
plt.imshow(dataMatrix[:,:,50])
#followinn commands do a false rgb conposition
dataMatrix = np.array(dataMatrix)
rgbMatrix = np.zeros((dataMatrix.shape[0], dataMatrix.shape[1], 3))
rgbMatrix[:, :, 0] = dataMatrix[:, :, 70]
rgbMatrix[:, :, 1] = dataMatrix[:, :, 50]
rgbMatrix[:, :, 2] = dataMatrix[:, :, 20]
rgbMatrix /= np.max(rgbMatrix)
plt.figure(figsize=[10, 10])
plt.imshow(rgbMatrix)
plt.show()

#transforms 'dataMatrix' nxnxd in (nxn)xd and assign to 'X' object (reshape() could be used instead also)
#X=[]
#for list1 in dataMatrix:
#    for list2 in list1:
#        X.append(list2)
#X = np.array(X)
#print(X)
#print(X.shape)
#np.save('C:/Users/Eduardo Kazuo Nakao/Tese/Implementação/Kennedy Space Center/KSC Matrizes/KSC',X)


"""loading, formating and showing classes matrix Y"""
classes = scipy.io.loadmat('C:/Users/Eduardo Kazuo Nakao/Tese/Implementação/Scenes/KSC_gt.mat')
print(classes)
classesMatrix = classes['KSC_gt']
print(classesMatrix)
print(classesMatrix.shape)
plt.imshow(classesMatrix, cmap=plt.cm.get_cmap('Spectral'))
plt.colorbar(ticks = range(0,14))
#transforms 'classesMatrix' nxn in (nxn)1 and assign to 'Y' object

#Y=[]
#for list1 in classesMatrix:
#    for list2 in list1:
#        Y.append(list2)
#Y = np.array(Y)
#np.save('C:/Users/Eduardo Kazuo Nakao/Tese/Implementação/Kennedy Space Center/KSC Matrizes/KSC_gt.npy',Y)
#print(Y)
#print(Y.shape)
##simple list with only the distinct values from 'Y'
#print(set(Y))

#swapping classes labels to be incremental in case image needs (necessary step for label matching function)
#setlisty = list(set(Y))
#for i in range(0,Y.size):
#    Y[i] = setlisty.index(Y[i])
#print(set(Y))
#np.save('C:/Users/Eduardo Kazuo Nakao/Salinas_corrected_gt_relabed.npy',Y)
#Y = np.load('C:/Users/Eduardo Kazuo Nakao/Salinas_corrected_gt_relabed.npy')
#print("Y relabed")
#print(Y)
#print(Y.shape)
#print(set(Y))