#Part 3.3: Function Approximation

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import sys
import math

numEpoch = 4000
eta = .01
alpha = .9

#globals
numHiddenNodes = int(sys.argv[1])

#3.3.1: generate points
x = np.arange(-5, 5.5, 0.5)
y = np.arange(-5, 5.5, 0.5)
X, Y = np.meshgrid(x, y)
X = X.flatten()
Y = Y.flatten()
targets = (np.exp(-X**2*0.1)* np.exp(-Y**2*0.1)) - 0.5
ones = np.ones(targets.shape[0])
patterns = np.array([X.flatten(), Y.flatten(), ones])

def createWeightsMatrix(xDim, yDim):
    return .1* np.random.randn(xDim, yDim)

def sigmoid_func(x):
  return (2/(1+math.exp(-x))) -1
sig_func = np.vectorize(sigmoid_func)

def deriv_sigmoid_func(x):
  return ((1+sigmoid_func(x))*(1-sigmoid_func(x)))/2
sig_func_deriv = np.vectorize(deriv_sigmoid_func)

#3.3.2
#initialize weights
weights = (createWeightsMatrix(numHiddenNodes, 3))
v = createWeightsMatrix(1, numHiddenNodes+1)
dw = np.zeros((numHiddenNodes, 3))
dv = np.zeros((1, numHiddenNodes+1))

outputs = [] #predicted z values at each iteration
MSEs = []
for i in range(0, numEpoch):
  #forward pass on training points
  hin = weights.dot(patterns)
  hout = sig_func(hin)
  hout = np.row_stack((hout, ones.T))

  oin = v.dot(hout)
  out = sig_func(oin)

  #backward pass
  delta_o = np.multiply( (out - targets), (sig_func_deriv(out)))
  delta_h = np.multiply( (v.T * delta_o),(sig_func_deriv(hout)))
  delta_h = delta_h[0:numHiddenNodes, :]
  
  #weight update
  dw = np.multiply(dw, alpha) - np.multiply((delta_h.dot(patterns.T)), (1-alpha))
  dv = np.multiply(dv, alpha) - np.multiply((delta_o.dot(hout.T)), (1-alpha))
  weights = weights + np.multiply(dw,eta)
  v = v + np.multiply(dv,eta)

#TESTING
  MSE = 0
  for i in range(len(out)):
    MSE += (out[:,i] - targets[i]) ** 2
  MSEs.append(MSE/len(out))

  outputs.append(out.flatten())

#create animation

def init():
    return

def update(frame):
    ax.clear()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Approximated Z')
    ax.set_title('Epoch ' + str(frame) + ' MSE '+str(MSEs[frame][0]))
    return ax.plot_trisurf(X, Y, outputs[int(frame)], cmap=cm.coolwarm)

fig = plt.figure(figsize = (21,21))
ax = fig.gca(projection='3d')
ani = FuncAnimation(fig, update, np.arange(0, numEpoch, 40), interval=200, init_func=init, save_count=1, blit=False)
ani.save(str(numHiddenNodes) + 'NodesAnimation.gif', dpi=80, writer='imagemagick')

# # Plot original function
# trainFig = plt.figure(figsize=(21,21))
# trainAx = trainFig.gca(projection='3d')
# trainAx.plot_trisurf(X, Y, targets, cmap=cm.coolwarm)
# trainAx.set_xlabel('x')
# trainAx.set_ylabel('y')
# trainAx.set_zlabel('f(x,y)')
# trainAx.set_title('Function Data')

# create mse plot
mseFig = plt.figure()
mseAx = mseFig.gca()
mseAx.scatter(range(numEpoch), MSEs)
mseAx.set_xlabel('Number of Epochs')
mseAx.set_ylabel('Mean Squared Error')
mseAx.set_title(str(numHiddenNodes) + ' Hidden Nodes')
mseAx.text(100,100,'Final MSE = ' + str(MSEs[numEpoch - 1]))
mseFig.savefig(str(numHiddenNodes) + 'NodesMse.png')

# save the plot of the last epoch
finalFig = plt.figure()
finalAx = finalFig.gca(projection='3d')
finalAx.set_xlabel('X')
finalAx.set_ylabel('Y')
finalAx.set_zlabel('Approximated Z')
finalAx.set_title(str(numHiddenNodes) + ' Hidden Nodes')
finalAx.plot_trisurf(X, Y, outputs[numEpoch - 1], cmap=cm.coolwarm)
finalFig.savefig(str(numHiddenNodes) + 'NodesFinal.png')

plt.show()
