#neural networks
#feedforward
#import required libraries
import numpy as np
#we will start with required input features
def weightsInitialization(inputFeatures, hidLayer, outLayer):
    #define the architecture of the network
    #lets suppose in our case we take a network of 1 input layer of 24 input units 3 hidden layers with units of 40, 50 and 30 with binary output
    #list in wich we will store these weights layers
    weights = []
    #so the required matrices will be of size 
    weights.append(np.random.normal(0, 0.5, (inputFeatures, hidLayer[0])))
    for i in range(len(hidLayer)-1):
        weights.append(np.random.normal(0, 0.5, (hidLayer[i], hidLayer[i+1])))
    weights.append(np.random.normal(0, 0.5, (hidLayer[len(hidLayer)-1], outLayer)))
    return weights
initialWeights = weightsInitialization(24, [40, 50, 30], 1)
for j in range(len(initialWeights)):
    print(initialWeights[j].shape)
def feedForward(weights):
    

