import numpy as np
from random import randint
import matplotlib.pyplot as plt
import math
import sys
import scipy as sp
import numpy.linalg as la

def sign(x):
  if x < 0:
    return -1
  else:
    return 1

def possibleVectors(n): # returns list of all possible binary vectors of length n
    if n == 0:
        return [[]]
    newVecs = []
    oldVecs = possibleVectors(n - 1)
    for v in oldVecs:
        newVecs.append(v + [1])
        newVecs.append(v + [-1])
    return newVecs

def calculateMatrix(x1, x2):
  return np.outer(x1, x2)

def createWeights(neu, training_data):
  w = np.zeros([neu, neu])
  #for data in training_data:
  #      w += calculateMatrix(data, data)
  tdata=np.asarray(training_data)
  w=np.matmul(tdata.T,tdata)
  for diag in range(neu):
    w[diag][diag] = 0
  return np.asarray(w)

def applyUpdate(weights, patternIn):
  pattern = np.copy(patternIn)
  pattern = np.matmul(weights,pattern)
  sign_vec=np.vectorize(sign)
  pattern=sign_vec(pattern)
  return pattern

def applySequentialUpdate(weights, patternIn, nodeNum):
  pattern = np.copy(patternIn)
  #for i in range(0, len(pattern)):
  sum = 0
  for j in range(0, len(pattern)):
    sum += (weights[nodeNum][j] * pattern[j])
  pattern[nodeNum]= sign(sum)
  return pattern

def trainUntilConverge(weights, training_data, max):
  inputOld = applyUpdate(weights, training_data)
  maxTrials = max
  tried = 0
#  reachedStableState = False
  while not (tried > maxTrials):
    inputNew = applyUpdate(weights, inputOld)
#    if np.array_equal(inputOld, inputNew):
#      reachedStableState = True
#      break
#    else:
    inputOld = inputNew
    tried += 1
  return np.asarray(inputNew)

def trainUntilConvergeSequential(weights, training_data, dim, max):
  maxTrials = max
  tried = 0
  nodeNum = randint(0, dim)
  inputOld = applySequentialUpdate(weights, training_data, nodeNum)
  while not (tried > maxTrials):
    nodeNum = randint(0, dim) #chooose random unit
    inputNew = applySequentialUpdate(weights, inputOld, nodeNum)
    inputOld = inputNew
    tried += 1
  return np.asarray(inputNew)

def trainUntilConvergeSequentialEnergy(weights, training_data, dim, max, step):
  maxTrials = max
  tried = 0
  E = []
  nodeNum = randint(0, dim)
  inputOld = applySequentialUpdate(weights, training_data, nodeNum)
  while not (tried > maxTrials):
    nodeNum = randint(0, dim) #chooose random unit
    inputNew = applySequentialUpdate(weights, inputOld, nodeNum)
    inputOld = inputNew
    if ((tried % step) == 0):
        E.append(energy(weights, inputOld))
    tried += 1
  return (np.asarray(inputNew), E)

#find number of attractors
def findAttractors(weights):
  attractors = []
  vecs = possibleVectors(8)
  for vec in vecs:
    distorted = trainUntilConverge(weights, vec, 10)
    distorted = distorted.tolist()
    attractors.append(distorted)
  attractors = [list(x) for x in set(tuple(x) for x in attractors)]
  return attractors

def createSparsePatterns(num, dim, rho):
  sp = np.zeros([num, dim])
  binary = np.array([0, 1])
  for i in range(0, num):
    sp[i] = np.random.choice(binary, dim, p=[1-rho, rho])
  #sp = np.asarray(sp)
  return sp

#TODO: this is copied fromCreateWeights, just add the rho parameter in the matmul operation
def createSparseWeights(neu, rho, training_data):
  w = np.zeros([neu, neu])
  #for data in training_data:
  #      w += calculateMatrix(data, data)
  tdata=np.asarray(training_data)
  for i in range(neu):
    for j in range(neu):
      for k in range(len(training_data)):
        w[i][j] += (training_data[k][i] - rho)*(training_data[k][j] - rho)
  for diag in range(neu):
    w[diag][diag] = 0
  return np.asarray(w)

def applySparseUpdate(weights, patternIn, bias):
  pattern = np.copy(patternIn)
  for i in range(len(pattern)):
    sum = 0
    for j in range(len(pattern)):
      sum += (weights[i, j] * pattern[j])
    sum -= bias
    pattern[i] = 0.5 + 0.5*sign(sum)
  return pattern

def findAverageActivity(num, dim, training_data):
  sum = 0;
  for i in range(1, num):
    for j in range(1, dim):
      sum += training_data[i, j]
  return sum*(1/(num*dim))

def testSparse(weights, bias, training_data):
  return np.asarray(applySparseUpdate(weights, training_data, bias))


#3.6
#bias terms can be any integer value
numPatterns = 30
numDimensions = 100
rho = 0.01
sparse = createSparsePatterns(numPatterns, numDimensions, rho)
sweights = createSparseWeights(numDimensions, rho, sparse)
biases = [0,0.2,0.4,0.6,0.8,1,1.5,2,2.5] 
for bias in biases:
    training_data = []
    percentStored = []
    nStored = []
    for i in range(numPatterns):
        numStored = 0
        training_data.append(sparse[i])
        w = createSparseWeights(numDimensions, rho, training_data)
        for j in range(i + 1):
            if (np.array_equal(testSparse(w,bias,sparse[j]), sparse[j])):
                numStored += 1
        nStored.append(numStored)
        percentStored.append(float(numStored)/(i+1))
    xaxis = np.arange(0,numPatterns)
    plt.plot(xaxis, percentStored)
    plt.title('Bias = '+str(bias))
    plt.show()
