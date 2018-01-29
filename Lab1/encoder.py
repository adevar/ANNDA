import numpy as np
import matplotlib.pyplot as plt
import math

#3.2.1
numEpoch = 60000
eta = .005
alpha = .9

#globals
numPointsPerCluster = 8
numHiddenNodes = 3
 
#create weights & setup
def createWeightsMatrix(xDim, yDim):
    return .1* np.random.randn(xDim, yDim)

weights = createWeightsMatrix(numHiddenNodes, 8+1)
v = createWeightsMatrix(8, numHiddenNodes+1)

patterns = -1*np.ones((8,numPointsPerCluster))
for i in range(0, numPointsPerCluster):
  patterns[i,i] = 1
some_ones=np.ones(8)
#targets = patterns
patterns=np.row_stack((patterns, some_ones))
targets = patterns[:-1]
print("patterns:" )
print(patterns)

def sigmoid_func(x):
  return (2/(1+math.exp(-x))) -1
sig_func = np.vectorize(sigmoid_func)

def deriv_sigmoid_func(x):
  return ((1+sigmoid_func(x))*(1-sigmoid_func(x)))/2
sig_func_deriv = np.vectorize(deriv_sigmoid_func)

dw = np.zeros((numHiddenNodes, 9))
dv = np.zeros((8, numHiddenNodes+1))

#batch
for i in range(0, numEpoch):
  #forward pass
  hin = weights.dot(patterns)
  hout = sig_func(hin)
  hout = np.row_stack((hout, some_ones.T))

  #print (hin.shape)
  #print (hout.shape)

  oin = v.dot(hout)
  out = sig_func(oin)
  #print (oin.shape)
  #print (out.shape)

  #backward pass
  delta_o = np.multiply( (out - targets), (sig_func_deriv(out)))
  #print(delta_o.shape)
  delta_h = np.multiply( np.dot(v.T, delta_o),(sig_func_deriv(hout)))
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
     if (targets[i] * out[i] < 0).any():
        #print("sad")
        wrong = wrong+ 1
  print(wrong)      

print weights
print v
