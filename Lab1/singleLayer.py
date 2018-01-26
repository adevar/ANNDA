import numpy as np
import matplotlib.pyplot as plt
#3.1.1

#generate data points
numPointsPerCluster = 100
mean = [-5, -5]
cov = [[3.5, 0], [0, 3.5]]
sample1x, sample1y = np.random.multivariate_normal(mean, cov, numPointsPerCluster).T
mean = [7,6]
cov = [[3.5, 0], [0, 3.5]]
sample2x, sample2y = np.random.multivariate_normal(mean, cov, numPointsPerCluster).T

#plot data points
plt.figure(1,figsize=(10,10))
plt.scatter(sample1x, sample1y, color="green")
plt.scatter(sample2x, sample2y, color="yellow")
plt.axis('equal')
#plt.show()

#combine data points
ones=np.ones(numPointsPerCluster)
negs=-1*np.ones(numPointsPerCluster)
sample1=np.column_stack((sample1x,sample1y,ones))
sample2=np.column_stack((sample2x,sample2y,negs))
allpoints=np.concatenate((sample1, sample2), axis=0)

#shuffle the data points
np.random.shuffle(allpoints)
patterns = allpoints[:,:2]
targets  = allpoints[:,2]
ones=np.ones(2*numPointsPerCluster)
patterns=np.column_stack((patterns,ones))
patterns = patterns.T

#3.1.2
#perceptron learning

#create weights & setup
def createWeightsMatrix(xDim, yDim):
    return .1* np.random.randn(xDim, yDim)

#weights = [-.5,.5, .7]
weights = np.ndarray.flatten(createWeightsMatrix(1, 3))
#print(weights)
eta = .001
numEpochs = 10

#sequential
for i in range(0, numEpochs):
  for i in range(0, 2*numPointsPerCluster):
    currOutput = np.dot(weights, patterns[:,i])
    if currOutput > 0 and targets[i] < 0:
        weights = weights - eta*patterns[:,i]
      #	weights[0]+=-eta*patterns[0][i]
    elif currOutput < 0 and targets[i] > 0:
        weights = weights + eta*patterns[:,i]
    #print(weights)
x = np.linspace(-20, 20, 1000)
plt.plot(x, (weights[0]*x + weights[2])/-weights[1], color = "purple")
plt.title('perceptron sequential')
plt.show()

#batch
weights = np.ndarray.flatten(createWeightsMatrix(1, 3))
for i in range(0, numEpochs):
  currOutput = np.dot(weights, patterns)
  for i in range(0, 2*numPointsPerCluster):
    if currOutput[i] > 0 and targets[i] < 0:
        weights = weights - eta*patterns[:,i]
        #	weights[0]+=-eta*patterns[0][i]
    elif currOutput[i]<0 and targets[i] >0:
        weights = weights + eta*patterns[:,i]
  #print(weights)
plt.figure(2, figsize=(10,10))  
plt.scatter(sample1x, sample1y, color="green")
plt.scatter(sample2x, sample2y, color="yellow")
plt.axis('equal')
x = np.linspace(-20, 20, 1000)
plt.plot(x, (weights[0]*x + weights[2])/-weights[1], color = "cyan") 
plt.title('perceptron batch')
plt.show()

#delta rule
weights = np.ndarray.flatten(createWeightsMatrix(1, 3))

#sequential
for i in range(0, numEpochs):
  for i in range(0, 2*numPointsPerCluster):
    currOutput = np.dot(weights, patterns[:,i])
    weights = weights - eta*patterns[:,i]*(currOutput-targets[i])
    #print(weights)
plt.figure(3, figsize=(10,10))  
plt.scatter(sample1x, sample1y, color="green")
plt.scatter(sample2x, sample2y, color="yellow")
plt.axis('equal')
x = np.linspace(-20, 20, 1000)
plt.plot(x, (weights[0]*x + weights[2])/-weights[1], color = "red") 
plt.title('delta rule sequential')
plt.show()

#batch
weights = np.ndarray.flatten(createWeightsMatrix(1, 3))
for i in range(0, numEpochs):
    currOutput = np.dot(weights, patterns)  
    for i in range(0, 2*numPointsPerCluster):
        weights = weights - eta*patterns[:,i]*(currOutput[i]-targets[i])
    #print(weights)
plt.figure(4, figsize=(10,10))  
plt.scatter(sample1x, sample1y, color="green")
plt.scatter(sample2x, sample2y, color="yellow")
plt.axis('equal')
x = np.linspace(-20, 20, 1000)
plt.plot(x, (weights[0]*x + weights[2])/-weights[1], color = "blue") 
plt.title('delta rule batch')
plt.show()

#TODO: plot learning curve for each variant 
#comparisons between all 4 based on number or ratio of misclassified examples at each epoch
#how quickly do the algorithms converge

#3.1.3
#TODO: all of it
