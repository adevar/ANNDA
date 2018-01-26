import numpy as np
import matplotlib.pyplot as plt
import math

#3.2.1
numEpoch = 1000
eta = .001
alpha = .9

#globals
numPointsPerCluster = 100
numHiddenNodes = 3

#generate data points
mean = [-5, -5]
cov = [[3.5, 0], [0, 3.5]]
sample1x, sample1y = np.random.multivariate_normal(mean, cov, numPointsPerCluster).T
mean = [7,6]
cov = [[3.5, 0], [0, 3.5]]
sample2x, sample2y = np.random.multivariate_normal(mean, cov, numPointsPerCluster).T
mean = [3, -2]
cov = [[3.5, 0], [0, 3.5]]
sample3x, sample3y = np.random.multivariate_normal(mean, cov, numPointsPerCluster).T

#generate test data points
mean = [-5, -5]
cov = [[3.5, 0], [0, 3.5]]
tsample1x, tsample1y = np.random.multivariate_normal(mean, cov, .1*numPointsPerCluster).T
mean = [7,6]
cov = [[3.5, 0], [0, 3.5]]
tsample2x, tsample2y = np.random.multivariate_normal(mean, cov, .1*numPointsPerCluster).T
mean = [3, -2]
cov = [[3.5, 0], [0, 3.5]]
tsample3x, tsample3y = np.random.multivariate_normal(mean, cov, .1*numPointsPerCluster).T

#plot data points
plt.figure(1,figsize=(10,10))
plt.scatter(sample1x, sample1y, color="green")
plt.scatter(sample2x, sample2y, color="blue")
plt.scatter(sample3x, sample3y, color="orange")
plt.axis('equal')
plt.show()

#combine data points
ones = np.ones(numPointsPerCluster)
negs = -1*np.ones(numPointsPerCluster)
sample1 = np.column_stack((sample1x, sample1y, ones))
sample2 = np.column_stack((sample2x, sample2y, ones))
sample3 = np.column_stack((sample3x, sample3y, negs))
allpoints = np.concatenate((sample1, sample2, sample3), axis=0)

#combine test data points
tsample1 = np.column_stack((tsample1x, tsample1y, ones))
tsample2 = np.column_stack((tsample2x, tsample2y, ones))
tsample3 = np.column_stack((tsample3x, tsample3y, negs))
tallpoints = np.concatenate((tsample1, tsample2, tsample3), axis=0)

#shuffle the data points
np.random.shuffle(allpoints)
patterns = allpoints[:,:2]
targets  = allpoints[:,2]
ones = np.ones(3*numPointsPerCluster)
patterns=np.column_stack((patterns,ones))
patterns = patterns.T

#shuffle the test data points
np.random.shuffle(tallpoints)
tatterns = tallpoints[:,:2]
ttargets  = tallpoints[:,2]
tones = np.ones(.3*numPointsPerCluster)
tatterns=np.column_stack((tatterns, tones))
tatterns = tatterns.T

#create weights & setup
def createWeightsMatrix(xDim, yDim):
    return .1* np.random.randn(xDim, yDim)

weights = (createWeightsMatrix(numHiddenNodes, 3))

def sigmoid_func(x):
  return (2/(1+math.exp(-x))) -1
sig_func = np.vectorize(sigmoid_func)

def deriv_sigmoid_func(x):
  return ((1+sigmoid_func(x))*(1-sigmoid_func(x)))/2
sig_func_deriv = np.vectorize(deriv_sigmoid_func)

v = createWeightsMatrix(1, numHiddenNodes+1)
dw = np.zeros((numHiddenNodes, 3))
dv = np.zeros((1, numHiddenNodes+1))

#batch
for i in range(0, numEpoch):
  #forward pass
  hin = weights.dot(patterns)
  hout = sig_func(hin)
  hout = np.row_stack((hout, ones.T))

  #print (hin.shape)
  #print (hout.shape)

  oin = v.dot(hout)
  out = sig_func(oin)
  #print (oin.shape)
  #print (out.shape)

  #backward pass
  delta_o = np.multiply( (out - targets), (sig_func_deriv(out)))
  #print(delta_o.shape)
  delta_h = np.multiply( (v.T * delta_o),(sig_func_deriv(hout)))
  #print(delta_h.shape)
  delta_h = delta_h[0:numHiddenNodes, :]
  #print(delta_h.shape)
  
  #weight update
  dw = np.multiply(dw, alpha) - np.multiply((delta_h.dot(patterns.T)), (1-alpha))
  dv = np.multiply(dv, alpha) - np.multiply((delta_o.dot(hout.T)), (1-alpha))
  weights = weights + np.multiply(dw,eta)
  v = v + np.multiply(dv,eta)
  
#TESTING---results = np.dot(v,np.row_stack((np.dot(weights, patterns),ones.T)))
  wrong = 0
  for i in range (len(targets)):
      if ((out[:,i]) * (targets[i]) < 0):
          wrong += 1
  print(wrong)
  
#TODO:
#do sequential (not anu)
#testing with test data, both sequential and batch
#analyze number of hidden nodes needed
#learning curves
#look at everything in spec in that section
