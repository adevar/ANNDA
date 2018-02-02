import numpy as np
import matplotlib.pyplot as plt
import math
import sys

#3.2.1
numEpoch = 2000
eta = .001
alpha = .9

#globals
numPointsPerCluster = 100
numTestPointsPerCluster = 50
numHiddenNodes = int(sys.argv[1])

#generate data points
mean = [-5, -5]
cov = [[3.5, 0], [0, 3.5]]
sample1x, sample1y = np.random.multivariate_normal(mean, cov, numPointsPerCluster).T
tsample1x, tsample1y = np.random.multivariate_normal(mean, cov, numTestPointsPerCluster).T
mean = [7,6]
cov = [[3.5, 0], [0, 3.5]]
sample2x, sample2y = np.random.multivariate_normal(mean, cov, numPointsPerCluster).T
tsample2x, tsample2y = np.random.multivariate_normal(mean, cov, numTestPointsPerCluster).T
mean = [3, -2]
cov = [[3.5, 0], [0, 3.5]]
sample3x, sample3y = np.random.multivariate_normal(mean, cov, numPointsPerCluster).T
tsample3x, tsample3y = np.random.multivariate_normal(mean, cov, numTestPointsPerCluster).T

#plot training data points
plt.figure(1,figsize=(10,10))
plt.scatter(sample1x, sample1y, color="green")
plt.scatter(sample2x, sample2y, color="blue")
plt.scatter(sample3x, sample3y, color="orange")
plt.axis('equal')
#plt.show()

#combine training data points
ones = np.ones(numPointsPerCluster)
negs = -1*np.ones(numPointsPerCluster)
sample1 = np.column_stack((sample1x, sample1y, ones))
sample2 = np.column_stack((sample2x, sample2y, ones))
sample3 = np.column_stack((sample3x, sample3y, negs))
allpoints = np.concatenate((sample1, sample2, sample3), axis=0)

#combine test data points
tones = np.ones(numTestPointsPerCluster)
tnegs = -1*np.ones(numTestPointsPerCluster)
tsample1 = np.column_stack((tsample1x, tsample1y, tones))
tsample2 = np.column_stack((tsample2x, tsample2y, tones))
tsample3 = np.column_stack((tsample3x, tsample3y, tnegs))
tallpoints = np.concatenate((tsample1, tsample2, tsample3), axis=0)

#shuffle the training data points
np.random.shuffle(allpoints)
patterns = allpoints[:,:2]
targets = allpoints[:,2]
ones = np.ones(3*numPointsPerCluster)
patterns = np.column_stack((patterns,ones)).T

#shuffle the test data points
#np.random.shuffle(tallpoints)
tpatterns = tallpoints[:,:2]
ttargets  = tallpoints[:,2]
tones = np.ones(3*numTestPointsPerCluster)
tpatterns = np.column_stack((tpatterns, tones)).T

#create weights & setup
def createWeightsMatrix(xDim, yDim):
    return .1* np.random.randn(xDim, yDim)

def sigmoid_func(x):
  return (2/(1+math.exp(-x))) -1
sig_func = np.vectorize(sigmoid_func)

def deriv_sigmoid_func(x):
  return ((1+sigmoid_func(x))*(1-sigmoid_func(x)))/2
sig_func_deriv = np.vectorize(deriv_sigmoid_func)

weights = (createWeightsMatrix(numHiddenNodes, 3))
v = createWeightsMatrix(1, numHiddenNodes+1)
dw = np.zeros((numHiddenNodes, 3))
dv = np.zeros((1, numHiddenNodes+1))

misclassifiedTrain = []
MSEsTrain = []
misclassifiedTest = []
MSEsTest = []
#batch learning
for i in range(numEpoch):
    wrong = 0
    MSE = 0
    for j in range(len(targets)):
        #forward pass
        hin = weights.dot(patterns[:,j])
        hout = sig_func(hin)
 #       hout = np.row_stack((hout, np.array([1])))
        hout = np.append(hout, 1)
        oin = v.dot(hout)
        out = sigmoid_func(oin)
        #backward pass
        delta_o = np.multiply((out - targets[j]), (deriv_sigmoid_func(out)))
        delta_h = np.multiply((v.T * delta_o).flatten(), (sig_func_deriv(hout)).flatten())
        delta_h = delta_h[0:numHiddenNodes]
        #weight update
        dw = np.multiply(dw, alpha) - np.multiply(np.outer(np.array([[delta_h]]),np.array((patterns[:,j]))), (1-alpha))
        dv = np.multiply(dv, alpha) - np.multiply(np.outer(np.array([[delta_o]]),np.array((hout.T))), (1-alpha))
        weights = weights + np.multiply(dw,eta)
        v = v + np.multiply(dv,eta)
      
        #evaluating on training data
        if ((out) * (targets[j]) < 0):
            wrong += 1
        MSE += (out - targets[j]) ** 2
    misclassifiedTrain.append(1.0*wrong/len(targets))
    MSEsTrain.append(1.0*MSE/len(targets))

    #forward pass on testing data
    thin = weights.dot(tpatterns)
    thout = sig_func(thin)
    thout = np.row_stack((thout, tones.T))
    toin = v.dot(thout)
    tout = sig_func(toin)
    
    #evaluating on testing data
    twrong = 0
    tMSE = 0
    for i in range(len(ttargets)):
        if ((tout[:,i]) * (ttargets[i]) < 0):
            twrong += 1
        tMSE += (tout[:,i] - ttargets[i]) ** 2
    misclassifiedTest.append(1.0*twrong/len(ttargets))
    MSEsTest.append(1.0*tMSE/len(ttargets))

#plot mse curve
mseFig = plt.figure()
mseAx = mseFig.gca()

mseAx.plot(range(numEpoch), MSEsTrain, color='green', label='Training Data')
mseAx.plot(range(numEpoch), MSEsTest, color='red', label='Testing Data')
mseAx.set_xlabel('Number of Epochs')
mseAx.set_ylabel('Mean Squared Error')
mseAx.set_title(str(numHiddenNodes) + ' Nodes Learning Curves')
mseAx.legend()
mseFig.savefig(str(numHiddenNodes) + 'NodesTraining.png')

#plot misclassification ratio curve
misclassifyFig = plt.figure()
misclassifyAx = misclassifyFig.gca()
misclassifyAx.plot(range(numEpoch), misclassifiedTrain, color='green', label='Training Data')
misclassifyAx.plot(range(numEpoch), misclassifiedTest, color='red', label='Testing Data')
misclassifyAx.set_xlabel('Number of Epochs')
misclassifyAx.set_ylabel('Ratio of Misclassified')
misclassifyAx.set_title(str(numHiddenNodes) + ' Nodes Learning Curves')
misclassifyAx.legend()
misclassifyFig.savefig(str(numHiddenNodes) + 'NodesMisclassified.png')

print(str(numHiddenNodes) + ' Nodes, ' + str(MSEsTest[-1]) + ' Test Inacc ' + str(MSEsTrain[-1]) + ' Train Inacc')
