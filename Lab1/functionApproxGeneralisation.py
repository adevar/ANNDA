import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import time
import math
import sys

#3.3.3
numEpoch = 3000
eta = .01
alpha = .9
numTrainingPoints = int(sys.argv[1])

#globals
numHiddenNodes = int(sys.argv[2])

#generate function data
x = np.arange(-5, 5.5, 0.5)
y = np.arange(-5, 5.5, 0.5)
X, Y = np.meshgrid(x, y)
X = X.flatten()
Y = Y.flatten()
targets = (np.exp(-X**2*0.1)* np.exp(-Y**2*0.1)) - 0.5

ones = np.ones(numTrainingPoints)
tones = np.ones(targets.shape[0] - numTrainingPoints)
allPoints = np.column_stack((X, Y, targets))

#shuffle points
np.random.shuffle(allPoints)

#split into training and testing set
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

weights = (createWeightsMatrix(numHiddenNodes, 3))
v = createWeightsMatrix(1, numHiddenNodes+1)
dw = np.zeros((numHiddenNodes, 3))
dv = np.zeros((1, numHiddenNodes+1))

outputs = [] #predicted z values at each iteration
MSEs = [] #MSE at each iteration

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
  outputs.append(tout.flatten())

# plot predicted training data points
# trainFig = plt.figure(figsize=(21,21))
# trainAx = trainFig.gca(projection='3d')
# trainAx.plot_trisurf(trainingPatterns[0].flatten(), trainingPatterns[1].flatten(), out.flatten(), cmap=cm.coolwarm)
# trainFig.show()

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

fig = plt.figure(figsize = (21,21))
ax = fig.gca(projection='3d')
ani = FuncAnimation(fig, update, np.arange(0, numEpoch, 200), interval=10, init_func=init, save_count=1, blit=False)

#plot mse over time
mseFig = plt.figure()
mseAx = mseFig.gca()
mseAx.scatter(range(numEpoch), MSEs)
mseAx.set_xlabel('Number of Epochs')
mseAx.set_ylabel('Mean Squared Error')
mseAx.set_title(str(numHiddenNodes) + ' Hidden Nodes, n = ' + str(numTrainingPoints))
mseFig.savefig(str(numHiddenNodes) + 'Nodes' + str(numTrainingPoints) + 'PointsMse.png')

#plot the prediction of the testing points at the last epoch
finalFig = plt.figure()
finalAx = finalFig.gca(projection='3d')
finalAx.set_xlabel('X')
finalAx.set_ylabel('Y')
finalAx.set_zlabel('Approximated Z')
finalAx.set_title(str(numHiddenNodes) + ' Hidden Nodes, n = ' + str(numTrainingPoints))
finalAx.plot_trisurf(testingPatterns[0], testingPatterns[1], outputs[numEpoch - 1], cmap=cm.coolwarm)
finalFig.savefig(str(numHiddenNodes) + 'Nodes' + str(numTrainingPoints) + 'Points.png')

plt.show()
print('n = ' + str(numTrainingPoints) + ', h = ' + str(numHiddenNodes) + ', MSE = ' + str(MSEs[numEpoch - 1][0]))
