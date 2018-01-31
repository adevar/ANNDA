import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import time
import math

#3.3
numEpoch = 40000
eta = .01
alpha = .9
numTrainingPoints = 5

#globals
numHiddenNodes = 5

#3.3.1
x = np.arange(-5, 5.5, 0.5)
y = np.arange(-5, 5.5, 0.5)
X, Y = np.meshgrid(x, y)
X = X.flatten()
Y = Y.flatten()
targets = (np.exp(-X**2*0.1)* np.exp(-Y**2*0.1)) - 0.5

ones = np.ones(numTrainingPoints)
tones = np.ones(targets.shape[0] - numTrainingPoints)
allPoints = np.column_stack((X, Y, targets))

np.random.shuffle(allPoints)

trainingPatterns = np.column_stack((allPoints[:numTrainingPoints, :2], ones)).T
trainingTargets = allPoints[:numTrainingPoints, 2]
testingPatterns = np.column_stack((allPoints[numTrainingPoints:, :2], tones)).T
testingTargets = allPoints[numTrainingPoints:, 2]

def createWeightsMatrix(xDim, yDim):
    return .1* np.random.randn(xDim, yDim)

def sigmoid_func(x):
  return (2/(1+math.exp(-x))) -1
sig_func = np.vectorize(sigmoid_func)

def deriv_sigmoid_func(x):
  return ((1+sigmoid_func(x))*(1-sigmoid_func(x)))/2
sig_func_deriv = np.vectorize(deriv_sigmoid_func)

#3.3.2
weights = (createWeightsMatrix(numHiddenNodes, 3))
v = createWeightsMatrix(1, numHiddenNodes+1)
dw = np.zeros((numHiddenNodes, 3))
dv = np.zeros((1, numHiddenNodes+1))

outputs = [] #predicted z values at each iteration
MSEs = []
#batch
for i in range(0, numEpoch):
  #forward pass on training points
  hin = weights.dot(trainingPatterns)
  hout = sig_func(hin)
  hout = np.row_stack((hout, ones.T))

  oin = v.dot(hout)
  out = sig_func(oin)

  #forward pass on testing points
  thin = weights.dot(testingPatterns)
  thout = sig_func(thin)
  thout = np.row_stack((thout, tones.T))

  toin = v.dot(thout)
  tout = sig_func(toin)

  #backward pass
  delta_o = np.multiply( (out - trainingTargets), (sig_func_deriv(out)))
  delta_h = np.multiply( (v.T * delta_o),(sig_func_deriv(hout)))
  delta_h = delta_h[0:numHiddenNodes, :]
  
  #weight update
  dw = np.multiply(dw, alpha) - np.multiply((delta_h.dot(trainingPatterns.T)), (1-alpha))
  dv = np.multiply(dv, alpha) - np.multiply((delta_o.dot(hout.T)), (1-alpha))
  weights = weights + np.multiply(dw,eta)
  v = v + np.multiply(dv,eta)

#TESTING
  MSE = 0
  for i in range(len(tout)):
    MSE += (tout[:,i] - testingTargets[i]) ** 2
  MSEs.append(MSE/len(tout))

#plot data points

#tout = out[0:21]
#ttargets = targets[0:21]
#plt.figure(1,figsize=(10,10))
#plt.scatter(Y[0], tout, color="green")
#plt.scatter(Y[0], ttargets, color="blue")
#plt.axis('equal')
#plt.show()


trainFig = plt.figure(figsize=(21,21))
trainAx = trainFig.gca(projection='3d')
trainAx.plot_trisurf(trainingPatterns[0].flatten(), trainingPatterns[1].flatten(), out.flatten(), cmap=cm.coolwarm)
#fig2.show()

#create animation

def init():
    return

def update(frame):
    ax.clear()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Approximated Z')
    ax.set_title('Epoch ' + str(frame) + ' MSE '+str(MSEs[frame][0]))
    return ax.plot_trisurf(testingPatterns[0], testingPatterns[1], outputs[int(frame)], cmap=cm.coolwarm)

mseFig = plt.figure()
mseAx = mseFig.gca()
mseAx.scatter(range(numEpoch), MSEs)

fig = plt.figure(figsize = (21,21))
ax = fig.gca(projection='3d')
ani = FuncAnimation(fig, update, np.arange(0, numEpoch, 200), interval=10, init_func=init, save_count=1, blit=False)

plt.show()
