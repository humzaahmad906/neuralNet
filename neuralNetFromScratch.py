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
    biases = []
    #so the required matrices will be of size 
    weights.append(np.random.normal(0, 0.5, (inputFeatures, hidLayer[0])))
    biases.append(np.random.normal(0, 0.5, hidLayer[0]))
    for i in range(len(hidLayer)-1):
        weights.append(np.random.normal(0, 0.5, (hidLayer[i], hidLayer[i+1])))
        biases.append(np.random.normal(0, 0.5, hidLayer[i+1]))
    weights.append(np.random.normal(0, 0.5, (hidLayer[len(hidLayer)-1], outLayer)))
    biases.append(np.random.normal(0, 0.5, outLayer))
    return [weights, biases]
initialWeights = weightsInitialization(24, [40, 50, 30], 1)
for j in range(len(initialWeights[0])):
    print(initialWeights[0][j].shape)
#as now we have initialized our network lets see if it will be able to predict anything
def feedForward(input, weights, biases):
    dummyInput = None
    for k in range(len(weights)):
        if k == 0:
            dummyInput = np.dot(input, weights[k])+biases[k]
            #relu implementation
            dummyInput = np.maximum(0, dummyInput)
        elif k != (len(weights)-1):
            dummyInput = np.dot(dummyInput, weights[k])+biases[k]
            dummyInput = np.maximum(0, dummyInput)
        else:
            dummyInput = np.dot(dummyInput, weights[k])+biases[k]
            #sigmoid implementation
            dummyInput = 1/(1+np.exp(-dummyInput))
    return dummyInput
#some dummy network to check if feedforward network is working
result = feedForward(np.ones((1,24), np.float64), initialWeights[0], initialWeights[1])
print(result)
            

        


    

