import random
import numpy as np


"""function to separate in list w the samples of X belonging to class w"""
def samplesByClass(parameterX, parameterY):
    parameterXlist = parameterX.tolist()
    parameterYlist = parameterY.tolist()
    classesMatrix = []
    #each position of the list will store a list: the first list is the elements of class 0, the second of class 1 and so on
    #for each element of parameterX, if this element belongs to wth class, it will be appended to the wth list and removed from parameterX
    #the corresponding position will be removed from parameterY also
    for c in set(parameterY):
        i = 0
        classElements = [] #create the wth list (the list of class w)
        for r in range(len(parameterYlist)):
            if parameterYlist[i] == c:
                classElements.append(parameterXlist.pop(i))
                parameterYlist.pop(i)
                i = i - 1
            i = i + 1
        classesMatrix.append(classElements)
    return classesMatrix


"""function to reduce large images by randomly sampling some percentage of the elements of each class"""
def bagging(classesMatrix,percentage):
    reducedX = []
    reducedY = []
    for i in range(len(classesMatrix)):
        aux = len(classesMatrix[i])
        for j in range(int(percentage*aux)):
            k = random.randint(0,(len(classesMatrix[i]))-1)
            reducedX.append(classesMatrix[i].pop(k))
            reducedY.append(i)
    return [reducedX, reducedY]


"""function to reduce only the 0 class"""
def backgroundBagging(classesMatrix,percentage):
    reducedX = []
    reducedY = []
    aux = len(classesMatrix[0])
    for j in range(int(percentage*aux)):
            k = random.randint(0,(len(classesMatrix[0]))-1)
            reducedX.append(classesMatrix[0].pop(k))
            reducedY.append(0)
    for i in range (1,len(classesMatrix)):
        aux = len(classesMatrix[i])
        for j in range(aux):
            reducedX.append(classesMatrix[i].pop())
            reducedY.append(i)
    return [reducedX, reducedY]


"""separating the samples of each class and doing bootstrap aggregating"""
X = np.load('')
Y = np.load('')

classesMatrix = samplesByClass(X,Y)
classesMatrix = np.array(classesMatrix)
np.save('C:/Users/Eduardo Kazuo Nakao/_classesMatrix.npy',classesMatrix)

for i in range(1,41):
    classesMatrix = np.load('C:/Users/Eduardo Kazuo Nakao/Botswana_classesMatrix.npy')
    classesMatrix = list(classesMatrix)
    reduced = backgroundBagging(classesMatrix,0.05)
    np.save('C:/Users/Eduardo Kazuo Nakao/Botswana_reducedX_execution'+str(i),reduced[0])
    np.save('C:/Users/Eduardo Kazuo Nakao/Botswana_reducedY_execution'+str(i),reduced[1])