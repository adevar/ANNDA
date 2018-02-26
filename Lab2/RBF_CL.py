import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import scipy as sp
import numpy.linalg as la

class RBF:
  center=0
  variance=0
  def __init__(self,center, variance):
    self.center = center
    self.variance = variance

  def gFunc(self,inputVec):
      output=np.zeros(inputVec.size)
      if inputVec.size == 1:
        dist=self.center-inputVec
        output=math.exp((-((dist)**2))/(2*self.variance))
        return output
      else:
        for i in range(inputVec.size):
          dist=self.center-inputVec[i]
          output[i]=math.exp((-((dist)**2))/(2*self.variance))
        return output

def makeRBFNodes(num, variance):
  #make num Guassian RBFs
  centers=np.linspace(0,2*math.pi, num)
  #centers = 2*math.pi*np.random.random(num)
  rbfnodes=np.zeros(centers.size, dtype=object)
  for i in range(centers.size):
    node=RBF(centers[i],variance)
    rbfnodes[i]=RBF(centers[i],variance)
  return rbfnodes

def makeRBFNodes2d(num, variance):
  #make num Guassian RBFs
  centersx=np.linspace(0,1, num)
  centersy=np.linspace(0,1, num)
  centers = np.row_stack((centersx, centersy))
  centers=centers.T
  #centers = np.asarray(centers)
  rbfnodes=np.zeros(len(centers), dtype=object)
  for i in range(len(centers)):
    node=RBF(centers[i],variance)
    rbfnodes[i]=RBF(centers[i],variance)
  return rbfnodes

def sin2x(x):
  return np.sin(2*x)

def square2x(x):
  y = np.sin(2*x)
  if y == 0:
    return 0
  elif y>0:
    return 1
  elif y<0:
    return -1


numRbfNodes= 40
numEpochs = 40000
variance = .2
rbfVector = makeRBFNodes(numRbfNodes,variance)
#roughly these are the steps
weights = .1* np.random.randn(numRbfNodes)
patterns = np.arange(0, 2*math.pi, 0.1)
patterns = patterns.T
sin2xVectorized = np.vectorize(sin2x)
square2xVectorized = np.vectorize(square2x)
targets =sin2xVectorized(patterns)
targets = targets.T

testdata = np.arange(0.05, 2*math.pi, 0.1)
testdata = testdata.T
testTargets =(sin2xVectorized(testdata)).T

def incrementalCL(patterns, numRbfNodes, rbfVector, eta, weights, numEpochs, targets, numEpochsCL):
  error = []
  for k in range(0, numEpochsCL):
    randVectIndex = np.random.uniform(0, len(patterns)-1)
    winnerValue = sys.maxint
    winnerIndex = 0
    for j in range(0, numRbfNodes):
      dist = math.fabs(patterns[randVectIndex]-rbfVector[j].center)
      #print(dist)
      contender = dist
      if contender < winnerValue:
        winnerIndex = j
    rbfVector[winnerIndex].center +=eta*(patterns[randVectIndex]-rbfVector[winnerIndex].center)
  for k in range(0, numEpochs):
    np.random.shuffle(patterns)
    sum=0
    for i in range(0, len(patterns)):
      phi = np.zeros(numRbfNodes)
      for j in range (0, numRbfNodes):
        phi[j] = rbfVector[j].gFunc(patterns[i])
      #print(phi.shape)
      f = (phi.T).dot(weights)
      deltaW = eta * (sin2x(patterns[i]) - f) * phi
      weights = weights + deltaW
      sum = sum + (np.absolute(sin2x(patterns[i]) - f))
      #print(np.average(np.absolute(sin2x(patterns[i]) - f)))
    print(sum/len(patterns))
    error.append(sum/len(patterns))
  return np.asarray(error)

def plotRbfNodes(patterns, rbfVector, numRbfNodes):
  patternsY = np.zeros(len(patterns))
  plt.scatter(patterns, patternsY, color='cyan', label='Input Data')
  rbfX = []
  for i in range(0, numRbfNodes):
    rbfX.append(rbfVector[i].center)
  rbfY = np.zeros(len(rbfX))
  plt.scatter(rbfX, rbfY, color = 'red', label = "RBF Nodes")
  plt.legend()
  plt.xlabel('x')
  #plt.title('')
  plt.show()
 
def incrementalCLLeaky(patterns, numRbfNodes, rbfVector, etaW, etaL, weights, numEpochs, targets, numEpochsCL):
  for k in range(0, numEpochsCL):
    randVectIndex = np.random.uniform(0, len(patterns)-1)
    winnerValue = sys.maxint
    winnerIndex = 0
    for j in range(0, numRbfNodes):
      dist = math.fabs(patterns[randVectIndex]-rbfVector[j].center)
      #print(dist)
      contender = dist
      if contender < winnerValue:
        winnerIndex = j
        winnerValue = contender
    for j in range(0, numRbfNodes):
      if j == winnerIndex:
        rbfVector[winnerIndex].center +=etaW*(patterns[randVectIndex]-rbfVector[winnerIndex].center)
      else:
        rbfVector[j].center+=etaL*(patterns[randVectIndex]-rbfVector[winnerIndex].center)
  for k in range(0, numEpochs):
    np.random.shuffle(patterns)
    sum=0
    for i in range(0, len(patterns)):
      phi = np.zeros(numRbfNodes)
      for j in range (0, numRbfNodes):
        phi[j] = rbfVector[j].gFunc(patterns[i])
      #print(phi.shape)
      f = (phi.T).dot(weights)
      deltaW = etaW * (sin2x(patterns[i]) - f) * phi
      weights = weights + deltaW
      sum = sum + (np.absolute(sin2x(patterns[i]) - f))
      #print(np.average(np.absolute(sin2x(patterns[i]) - f)))
    print(sum/len(patterns))
  return weights

