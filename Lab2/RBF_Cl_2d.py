import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import scipy as sp
import numpy.linalg as la

def incrementalCLLeaky2D(patterns, numRbfNodes, rbfVector, etaW, etaL, weights, numEpochs, targets, numEpochsCL):
  #patterns = np.asarray(patterns)
  #rbfVector = np.asarray(rbfVector)
  targets = np.reshape(targets, (100, 2))
  nodesX=[]
  nodesY=[]
  for k in range(0, numEpochsCL):
    randVectIndex = np.random.uniform(0, len(patterns)-1)
    randVectIndex = int(randVectIndex)
    winnerValue = sys.maxint
    winnerIndex = 0
    #print("patterns aka input shape is: ",patterns.shape)
    for j in range(0, numRbfNodes):
      #print(patterns[randVectIndex])
      dist =la.norm(np.asarray(patterns[randVectIndex]-rbfVector[j].center),2)
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
  for j in range(0, numRbfNodes):
    nodesX.append(rbfVector[j].center[0])
    nodesY.append(rbfVector[j].center[1])
  plt.figure()
  patterns = patterns.T
  plt.scatter(patterns[0], patterns[1], color='cyan')
  plt.scatter(nodesX, nodesY, color='red')
  plt.title('Position of RBF nodes in input space')
  plt.xlabel('Angle')
  plt.ylabel('Velocity')
  plt.show()
  patterns = patterns.T
  errorVect = []
  for k in range(0, numEpochs):
    #np.random.shuffle(patterns)
    sum=0
    for i in range(0, len(patterns)):
      phi = np.zeros(numRbfNodes, dtype=float)
      for j in range (0, numRbfNodes):
        phi[j] = rbfVector[j].gFunc(patterns[i])
        #print(phi[j])
      phi = np.reshape(phi, (numRbfNodes, 1))
      #print("phi shape:", phi.shape)
      f = phi.T.dot(weights)
      #hi=np.asarray(targets[1]).dot(f))
      #print("targets shape: ",targets.shape)
      #print("f shape", f.shape)
      #print("targets shape", targets.shape)
      temp = targets[i]
      #print("temp shape: ",temp.shape)
      #print(phi)
      arr=np.zeros((2,1))
      #print("arr shape: ",arr.shape)
      sub=temp-f
      #print("sub shape: ",sub.shape)
      dotted=(phi).dot(sub)
      deltaW = etaW * dotted
      weights = weights + deltaW
      sum = sum + la.norm((targets[i] - f), 2)
      #sum = sum + (np.absolute(sin2x(patterns[i]) - f))
      #print(np.average(np.absolute(sin2x(patterns[i]) - f)))
    print(sum/len(patterns))
    errorVect.append(sum/len(patterns))
  xvals=np.linspace(1,numEpochs+1, numEpochs)
  plt.plot(xvals, errorVect, color='cyan', label='Training Data Error')
  plt.legend()
  plt.xlabel('Epoch')
  plt.ylabel('Absolute Residual Error')
  #plt.ylim(-1.1,1.1)
  plt.xlim(1, numEpochs)
  plt.title('Training data from ballistical data')
  plt.show()
  return weights


def testIncremental2d(numRbfNodes, rbfVector,  weights, numEpochs, testdata, testTargets):
  #test with one run through batch
  phiMatrix=[]
  for j in range(0, numRbfNodes):
    column = []
    for k in range(0, 100):
      column.append(rbfVector[j].gFunc(testdata[k]))
    #column=rbfVector[j].gFunc(testdata)
    #print(column)
    phiMatrix.append(column)
  phiMatrix=np.asarray(phiMatrix)
  print("phi:", phiMatrix.shape)
  testoutput=(phiMatrix.T).dot(weights)
  print("test output:", testoutput.shape)
  testError = 0
  for j in range(0, 100):
    testError += la.norm(testoutput[j] - testTargets[j], 2)
  #print(testError)
  return testError/100


angles = []
velocities = []
distances = []
heights = []

with open("ballist.dat", "r+") as f:
  data = f.readlines()
  for line in data:
    words = line.split()
    angles.append(float(words[0]))
    velocities.append(float(words[1]))
    distances.append(float(words[2]))
    heights.append(float(words[3]))
angles=np.asarray(angles)
velocities=np.asarray(velocities)
heights=np.asarray(heights)
distances=np.asarray(distances)

inputs = np.column_stack((angles, velocities))
#inputs=inputs.T
outputs = np.column_stack((distances, heights))
#output=outputs.T
#print(inputs)
#print(inputs.shape)

rbfVector = makeRBFNodes2d(numRbfNodes,variance)
#print(rbfVector.shape)
weightsX = .1* np.random.randn(numRbfNodes)
weightsY = .1* np.random.randn(numRbfNodes)
weights = np.row_stack((weightsX, weightsY))
weights=weights.T
print("weights shape", weights.shape)
w = incrementalCLLeaky2D(inputs, numRbfNodes, rbfVector, .02, .002, weights, numEpochs, outputs, 200)
