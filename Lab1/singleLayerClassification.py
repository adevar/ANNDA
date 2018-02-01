import numpy as np
import matplotlib.pyplot as plt
import time
#3.1.1

#globals for linearly separable
eta = .001
numEpochs = 7

#generate data points
numPointsPerCluster = 100
mean = [-5, -5]
cov = [[3.5, 0], [0, 3.5]]
sample1x, sample1y = np.random.multivariate_normal(mean, cov, numPointsPerCluster).T
mean = [7,6]
cov = [[3.5, 0], [0, 3.5]]
sample2x, sample2y = np.random.multivariate_normal(mean, cov, numPointsPerCluster).T


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

#create weights & setup
def createWeightsMatrix(xDim, yDim):
    return .1* np.random.randn(xDim, yDim)

#3.1.2
#perceptron learning

#weights = [-.5,.5, .7]
weights = np.ndarray.flatten(createWeightsMatrix(1, 3))
#print(weights)

#sequential
psepochs = []
pswrong = []
for j in range(0, numEpochs):
  wrong = 0
  for i in range(0, 2*numPointsPerCluster):
    currOutput = np.dot(weights, patterns[:,i])
    if currOutput > 0 and targets[i] < 0:
        weights = weights - eta*patterns[:,i]
        wrong+=1
      # weights[0]+=-eta*patterns[0][i]
    elif currOutput < 0 and targets[i] > 0:
        weights = weights + eta*patterns[:,i]
        wrong+=1
    #print(weights)
  psepochs.append(j)
  pswrong.append(wrong)

#plot data points
plt.figure(1,figsize=(10,10))
plt.scatter(sample1x, sample1y, color="green")
plt.scatter(sample2x, sample2y, color="yellow")
plt.axis('equal')
x = np.linspace(-20, 20, 1000)
plt.plot(x, (weights[0]*x + weights[2])/-weights[1], color = "red")
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.title('linearly separable perceptron sequential')
plt.show()

#plot epoch vs error
psepochs = np.asarray(psepochs)
pswrong = np.asarray(pswrong)

#plt.figure(2, figsize=(10,10))
#plt.plot(psepochs, pswrong, color="red")
#plt.title('linearly separable perceptron sequential learning error')
#plt.show

#batch
pbepochs = []
pbwrong = []
weights = np.ndarray.flatten(createWeightsMatrix(1, 3))
for j in range(0, numEpochs):
  wrong = 0
  currOutput = np.dot(weights, patterns)
  for i in range(0, 2*numPointsPerCluster):
    if currOutput[i] > 0 and targets[i] < 0:
        weights = weights - eta*patterns[:,i]
        wrong+=1
        # weights[0]+=-eta*patterns[0][i]
    elif currOutput[i]<0 and targets[i] >0:
        weights = weights + eta*patterns[:,i]
        wrong+=1
  #print(weights)
  pbepochs.append(j)
  pbwrong.append(wrong)

plt.figure(2, figsize=(10,10))
plt.scatter(sample1x, sample1y, color="green")
plt.scatter(sample2x, sample2y, color="yellow")
plt.axis('equal')
x = np.linspace(-20, 20, 1000)
plt.plot(x, (weights[0]*x + weights[2])/-weights[1], color = "cyan") 
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.title('linearly separable perceptron batch')
plt.show()

pbepochs = np.asarray(pbepochs)
pbwrong = np.asarray(pbwrong)

plt.figure(3, figsize=(10,10))
plt.plot(psepochs, pswrong, color="red")
plt.plot(pbepochs, pbwrong, color="cyan")
plt.title('linearly separable perceptron learning error')
plt.show

#delta rule
weights = np.ndarray.flatten(createWeightsMatrix(1, 3))

#sequential
dsepochs = []
dserror = []
for j in range(0, numEpochs):
  MSE = 0
  for i in range(0, 2*numPointsPerCluster):
    currOutput = np.dot(weights, patterns[:,i])
    weights = weights - eta*patterns[:,i]*(currOutput-targets[i])
    #mean-squared error calculation
    MSE = MSE + (targets[i]-currOutput)*(targets[i]-currOutput)
    #print(weights)
  dserror.append(MSE/(2*numPointsPerCluster))
  dsepochs.append(j)

plt.figure(4, figsize=(10,10))
plt.scatter(sample1x, sample1y, color="green")
plt.scatter(sample2x, sample2y, color="yellow")
plt.axis('equal')
x = np.linspace(-20, 20, 1000)
plt.plot(x, (weights[0]*x + weights[2])/-weights[1], color = "red") 
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.title('linearly separable delta rule sequential')
plt.show()

dsepochs = np.asarray(dsepochs)
dserror = np.asarray(dserror)

# plt.figure(6, figsize=(10,10))
# plt.plot(dsepochs, dserror, color="red")
# plt.title('linearly separable delta rule sequential learning error')
# plt.show

#batch
weights = np.ndarray.flatten(createWeightsMatrix(1, 3))

dbepochs = []
dberror = []
for j in range(0, numEpochs):
  MSE = 0
  currOutput = np.dot(weights, patterns)  
  for i in range(0, 2*numPointsPerCluster):
    weights = weights - eta*patterns[:,i]*(currOutput[i]-targets[i])
    #mean-squared error calculation
    #MSE = MSE + (targets[i]-currOutput[i])*(targets[i]-currOutput[i])
    #print(weights)
  temp = (targets-currOutput)
  temp = np.dot(temp, temp.T)
  MSE = MSE + temp/temp.size
  dberror.append(MSE/(2*numPointsPerCluster))
  dbepochs.append(j)

plt.figure(5, figsize=(10,10))
plt.scatter(sample1x, sample1y, color="green")
plt.scatter(sample2x, sample2y, color="yellow")
plt.axis('equal')
x = np.linspace(-20, 20, 1000)
plt.plot(x, (weights[0]*x + weights[2])/-weights[1], color = "cyan")
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.title('linearly separable delta rule batch')
plt.show()

dbepochs = np.asarray(dbepochs)
dberror = np.asarray(dberror)

# plt.figure(8, figsize=(10,10))
# plt.plot(dsepochs, dserror, color="red")
# plt.title('linearly separable delta rule batch learning error')
# plt.show

#print (dbepochs)
#print (dberror)

#plt.figure(6, figsize=(10,10))
#plt.plot(dsepochs, dserror, color="red")
#plt.title('linearly separable delta rule learning error SEQ')
#plt.show
plt.figure(20, figsize=(10,10))
plt.plot(dsepochs, dserror, color="red")
plt.plot(dbepochs, dberror, color="cyan")
plt.xlim(0, 5)
plt.ylim(0, .2)
plt.title('linearly separable delta rule learning error')
plt.show

#TODO: plot learning curve for each variant 
#comparisons between all 4 based on number or ratio of misclassified examples at each epoch
#how quickly do the algorithms converge

#3.1.3

#globals for non-linearly separable
eta = .001
numEpochs = 40

#generate data points
numPointsPerCluster = 100
mean = [-5, -5]
cov = [[5, 0], [0, 5]]
sample1x, sample1y = np.random.multivariate_normal(mean, cov, numPointsPerCluster).T
mean = [-3, -3]
cov = [[5, 0], [0, 5]]
sample2x, sample2y = np.random.multivariate_normal(mean, cov, numPointsPerCluster).T

# plt.figure(1,figsize=(10,10))
# plt.scatter(sample1x, sample1y, color="green")
# plt.scatter(sample2x, sample2y, color="blue")
# plt.axis('equal')
# plt.show()

#combine data points
ones = np.ones(numPointsPerCluster)
negs = -1*np.ones(numPointsPerCluster)
sample1 = np.column_stack((sample1x, sample1y, ones))
sample2 = np.column_stack((sample2x, sample2y, ones))
allpoints = np.concatenate((sample1, sample2), axis=0)

#shuffle the data points
np.random.shuffle(allpoints)
patterns = allpoints[:,:2]
targets  = allpoints[:,2]
ones=np.ones(2*numPointsPerCluster)
patterns=np.column_stack((patterns,ones))
patterns = patterns.T

#weights = [-.5,.5, .7]
weights = np.ndarray.flatten(createWeightsMatrix(1, 3))
#print(weights)

#sequential
psepochs = []
pswrong = []
for j in range(0, numEpochs):
  wrong = 0
  for i in range(0, 2*numPointsPerCluster):
    currOutput = np.dot(weights, patterns[:,i])
    if currOutput > 0 and targets[i] < 0:
        weights = weights - eta*patterns[:,i]
        wrong+=1
      # weights[0]+=-eta*patterns[0][i]
    elif currOutput < 0 and targets[i] > 0:
        weights = weights + eta*patterns[:,i]
        wrong+=1
    #print(weights)
  psepochs.append(j)
  pswrong.append(wrong)

plt.figure(7, figsize=(10,10))
plt.scatter(sample1x, sample1y, color="green")
plt.scatter(sample2x, sample2y, color="yellow")
plt.axis('equal')
x = np.linspace(-20, 20, 1000)
plt.plot(x, (weights[0]*x + weights[2])/-weights[1], color = "red")
#plt.xlim(-10, 10)
#plt.ylim(-10, 10)
plt.title('non-linearly separable perceptron sequential')
plt.show()

#plot epoch vs error
psepochs = np.asarray(psepochs)
pswrong = np.asarray(pswrong)

# plt.figure(10, figsize=(10,10))
# plt.plot(psepochs, pswrong, color="red")
# plt.title('non-linearly separable perceptron sequential learning error')
# plt.show

#batch
weights = np.ndarray.flatten(createWeightsMatrix(1, 3))

pbepochs = []
pbwrong = []
for j in range(0, numEpochs):
  wrong = 0
  currOutput = np.dot(weights, patterns)
  for i in range(0, 2*numPointsPerCluster):
    if currOutput[i] > 0 and targets[i] < 0:
        weights = weights - eta*patterns[:,i]
        wrong+=1
        # weights[0]+=-eta*patterns[0][i]
    elif currOutput[i]<0 and targets[i] >0:
        weights = weights + eta*patterns[:,i]
        wrong+=1
  #print(weights)
  pbepochs.append(j)
  pbwrong.append(wrong)

plt.figure(8, figsize=(10,10))  
plt.scatter(sample1x, sample1y, color="green")
plt.scatter(sample2x, sample2y, color="yellow")
plt.axis('equal')
x = np.linspace(-20, 20, 1000)
plt.plot(x, (weights[0]*x + weights[2])/-weights[1], color = "cyan") 
#plt.xlim(-10, 10)
#plt.ylim(-10, 10)
plt.title('non-linearly separable perceptron batch')
plt.show()

pbepochs = np.asarray(pbepochs)
pbwrong = np.asarray(pbwrong)

plt.figure(9, figsize=(10,10))
plt.plot(psepochs, pswrong, color="red")
plt.plot(pbepochs, pbwrong, color="cyan")
plt.title('non-linearly separable perceptron learning error')
plt.show

#delta rule
weights = np.ndarray.flatten(createWeightsMatrix(1, 3))

#sequential
dsepochs = []
dserror = []
for j in range(0, numEpochs):
  MSE = 0
  for i in range(0, 2*numPointsPerCluster):
    currOutput = np.dot(weights, patterns[:,i])
    weights = weights - eta*patterns[:,i]*(currOutput-targets[i])
    #mean-squared error calculation
    MSE += (targets[i]-currOutput)*(targets[i]-currOutput)
    #print(weights)
  dserror.append(MSE/(2*numPointsPerCluster))
  dsepochs.append(j)

plt.figure(10, figsize=(10,10))  
plt.scatter(sample1x, sample1y, color="green")
plt.scatter(sample2x, sample2y, color="yellow")
plt.axis('equal')
x = np.linspace(-20, 20, 1000)
plt.plot(x, (weights[0]*x + weights[2])/-weights[1], color = "red") 
#plt.xlim(-10, 10)
#plt.ylim(-10, 10)
plt.title('non-linearly separable delta rule sequential')
plt.show()

dsepochs = np.asarray(dsepochs)
dserror = np.asarray(dserror)

# plt.figure(14, figsize=(10,10))
# plt.plot(dsepochs, dserror, color="red")
# plt.title('non-linearly separable delta rule sequential learning error')
# plt.show

#batch
weights = np.ndarray.flatten(createWeightsMatrix(1, 3))

dbepochs = []
dberror = []
for j in range(0, numEpochs):
  MSE = 0
  currOutput = np.dot(weights, patterns)  
  for i in range(0, 2*numPointsPerCluster):
    weights = weights - eta*patterns[:,i]*(currOutput[i]-targets[i])
    #mean-squared error calculation
    MSE += (targets[i]-currOutput[i])*(targets[i]-currOutput[i])
    #print(weights)
  dberror.append(MSE/(2*numPointsPerCluster))
  dbepochs.append(j)

plt.figure(11, figsize=(10,10))  
plt.scatter(sample1x, sample1y, color="green")
plt.scatter(sample2x, sample2y, color="yellow")
plt.axis('equal')
x = np.linspace(-20, 20, 1000)
plt.plot(x, (weights[0]*x + weights[2])/-weights[1], color = "cyan")
#plt.xlim(-10, 10)
#plt.ylim(-10, 10)
plt.title('non-linearly separable delta rule batch')
plt.show()

dbepochs = np.asarray(dbepochs)
dberror = np.asarray(dberror)

# plt.figure(16, figsize=(10,10))
# plt.plot(dsepochs, dserror, color="red")
# plt.title('non-linearly separable delta rule batch learning error')
# plt.show

plt.figure(12, figsize=(10,10))
plt.plot(dsepochs, dserror, color="red")
plt.plot(dbepochs, dberror, color="cyan")
plt.title('non-linearly separable delta rule learning error')
plt.show


