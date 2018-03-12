import numpy as np
from random import randint
import random
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

def flip(v, n):
    vec = np.copy(v)
    chosen = random.sample(range(0,len(vec)), n)
    for a in chosen:
        vec[a] *= -1
    return vec

def calculateMatrix(x1, x2):
  return np.outer(x1, x2)

def createWeights(neu, training_data):
  w = np.zeros([neu, neu])
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

def test(weights, training_data):
  return np.asarray(applyUpdate(weights, training_data))

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

def energy(weights, pattern):
    E = 0
    for i in range(0, len(pattern)):
        for j in range(0, len(pattern)):
            E -= weights[i, j] * pattern[i] * pattern[j]
    return E

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

##2.2
numNeurons = 8
x1 = [-1,-1,1,-1,1,-1,-1,1]
x2 = [-1,-1,-1,-1,-1,1,-1,-1]
x3 = [-1,1,1,-1,-1,1,-1,1]
x = [[], x1, x2, x3]
training_data = [x1, x2, x3]
eightWeights = createWeights(numNeurons,training_data)

xOut = [[]]
for i in range(1, 4):
    xOut.append(test(eightWeights, x[i]))
    print("Is x", i,  " a fixed point? ",np.array_equal(x[i], xOut[i]))

#3.1
x1d = [ 1, -1, 1, -1, 1, -1, -1, 1]
x2d = [ 1, 1, -1, -1, -1, 1, -1, -1]
x3d = [ 1, 1, 1, -1, 1, 1, -1, 1]
xd = [[], x1d, x2d, x3d]

newDistort = [[]]
for i in range(1, 4):
    newDistort.append(trainUntilConverge(eightWeights, xd[i], 10))

print("Did all the patterns converge towards stored patterns?")
for i in range(1, 4):
    print("x", i, " ", np.array_equal(x[i], newDistort[i]))


print("Attractors for 8 neuron are: ")
patternAttractors = findAttractors(eightWeights)
print(patternAttractors)

  
#what happens when starting pattern is more than half wrong
x4d = [-1,-1,-1,1,-1,1,1,-1]
x4dOut = trainUntilConverge(eightWeights, x4d, 10)
print("Was input with more than half of values distorted able to converge to the correct pattern?", np.array_equal(x[1], x4d))
print("In other words, it successfully converged to an attractor, though not the same attractor as x1")

file = open("pict.dat",'r')
pAll = (file.readline()).split(",")
p = [[]]
for i in range(11):
    p.append(pAll[(1024 * i):(1024 * (i + 1))])
    p[i + 1] = map(int, p[i + 1])
picWeights = createWeights(1024,[p[1], p[2], p[3]])
picWeights = np.asarray(picWeights)


# 3.3
# energy of attractors
normpicweights = picweights/(3.0)
for i in range(1,4):
    e = energy(normpicweights, p[i])
    print("energy of attractor p", i, ": ", e)

# energy of distorted patterns
for i in range(10,12):
    e = energy(normpicweights, p[i])
    print("energy of distorted pattern p", i, ": ", e)

# energy changing over time
p10out, p10oute = trainuntilconvergesequentialenergy(normpicweights, p[10], 1023, 10000, 100)
xaxis = np.arange(0, 10001, 100)
plt.plot(xaxis, p10oute)
plt.title('energy as p10 approaches p1')
plt.xlabel('iteration')
plt.ylabel('energy')
plt.show()
plt.figure(1)
plt.imshow(np.reshape((np.asarray(p10out)), (32,32)), cmap='greys')

p10Out, p10OutE = trainUntilConvergeSequentialEnergy(normPicWeights, p[11], 1023, 10000, 100)
xaxis = np.arange(0, 10001, 100)
plt.plot(xaxis, p10OutE)
plt.title('Energy as p11 Iterates')
plt.xlabel('Iteration')
plt.ylabel('Energy')
plt.show()
plt.figure(2)
plt.imshow(np.reshape((np.asarray(p10Out)), (32,32)), cmap='Greys')

# Random weights
randWeights = np.random.normal(size=(1024,1024))
for i in range(1024):
    randWeights[i,i] = 0
p1Out, p1OutE = trainUntilConvergeSequentialEnergy(randWeights, p[1], 1023, 10000, 100)
xaxis = np.arange(0, 10001, 100)
plt.plot(xaxis, p1OutE)
plt.title('Energy of p1 With Random Weights')
plt.xlabel('Iteration')
plt.ylabel('Energy')
plt.show()
plt.figure(3)
plt.imshow(np.reshape((np.asarray(p1Out)), (32,32)), cmap='Greys')

# Symmetric random weight matrix
randSymmetricWeights = 0.5 * (randWeights + randWeights.T)
p1Out, p1OutE = trainUntilConvergeSequentialEnergy(randSymmetricWeights, p[1], 1023, 10000, 100)
xaxis = np.arange(0, 10001, 100)
plt.plot(xaxis, p1OutE)
plt.title('Energy of p1 With Symmetric Random Weights')
plt.xlabel('Iteration')
plt.ylabel('Energy')
plt.show()
plt.figure(4)
plt.imshow(np.reshape((np.asarray(p1Out)), (32,32)), cmap='Greys')

# 3.4 Noise removal

# flip N percent of the bits
converged = []
convergedi = [[],[],[],[]]
for N in range(0,101):
    total = 0
    for i in range(1,4):
        noisy = flip(p[i], int(10.24 * N))
        noisyOut = trainUntilConverge(picWeights, noisy, 1)
        succ = np.array_equal(noisyOut, p[i])
        convergedi[i].append(1*succ)
        total += succ
    converged.append(total/3.)
xaxis = np.arange(0, 101)

plt.figure()
plt.suptitle('Noise Removal after a Single Step')

plt.subplot(2,2,1)
plt.plot(xaxis, convergedi[1], 'r*', label='p1')
plt.title('P1')
plt.ylabel('Successful Convergence?')

plt.subplot(2,2,2)
plt.plot(xaxis, convergedi[2], 'g*', label='p1')
plt.title('P2')
plt.ylabel('Successful Convergence?')

plt.subplot(2,2,3)
plt.plot(xaxis, convergedi[3], 'b*', label='p1')
plt.title('P3')
plt.xlabel('Percentage Noise')
plt.ylabel('Successful Convergence?')

plt.subplot(2,2,4)
plt.plot(xaxis, converged, 'k*', label='p1')
plt.title('Rate over P1, P2, P3')
plt.xlabel('Percentage Noise')
plt.ylabel('Successful Convergence?')
plt.show()

converged = []
convergedi = [[],[],[],[]]
for N in range(0,101):
    total = 0
    for i in range(1,4):
        noisy = flip(p[i], int(10.24 * N))
        noisyOut = trainUntilConverge(picWeights, noisy, 10)
        succ = np.array_equal(noisyOut, p[i])
        convergedi[i].append(1*succ)
        total += succ
    converged.append(total/3.)
xaxis = np.arange(0, 101)

plt.figure()
plt.suptitle('Noise Removal after 10 Steps')

plt.subplot(2,2,1)
plt.plot(xaxis, convergedi[1], 'r*', label='p1')
plt.title('P1')
plt.ylabel('Successful Convergence?')

plt.subplot(2,2,2)
plt.plot(xaxis, convergedi[2], 'g*', label='p1')
plt.title('P2')
plt.ylabel('Successful Convergence?')

plt.subplot(2,2,3)
plt.plot(xaxis, convergedi[3], 'b*', label='p1')
plt.title('P3')
plt.xlabel('Percentage Noise')
plt.ylabel('Successful Convergence?')

plt.subplot(2,2,4)
plt.plot(xaxis, converged, 'k*', label='p1')
plt.title('Rate over P1, P2, P3')
plt.xlabel('Percentage Noise')
plt.ylabel('Successful Convergence?')
plt.show()
