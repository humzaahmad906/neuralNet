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
    layerOutputsBefore = []
    layerOutputsAfter = []
    dummyInput = None
    for k in range(len(weights)):
        if k == 0:
            dummyInput = np.dot(input, weights[k])+biases[k]
            layerOutputsBefore.append(dummyInput)
            #relu implementation
            dummyInput = np.maximum(0, dummyInput)
            layerOutputsAfter.append(dummyInput)
        elif k != (len(weights)-1):
            dummyInput = np.dot(dummyInput, weights[k])+biases[k]
            layerOutputsBefore.append(dummyInput)
            dummyInput = np.maximum(0, dummyInput)
            layerOutputsAfter.append(dummyInput)
        else:
            dummyInput = np.dot(dummyInput, weights[k])+biases[k]
            layerOutputsBefore.append(dummyInput)
            #sigmoid implementation
            dummyInput = 1/(1+np.exp(-dummyInput))
            layerOutputsAfter.append(dummyInput)
    return [layerOutputsBefore, layerOutputsAfter]
#some dummy network to check if feedforward network is working
[layersResultBefore, layersResultAfter] = feedForward(np.ones((1,24), np.float64), initialWeights[0], initialWeights[1])
lr = 0.1
#backpropogation
#define energy function
#energy function derivative
gT = 1
Oout = layersResultAfter.pop()
dEbydOout = -1*(gT/Oout)+(1-gT)/(1-Oout)
#sigmoid derivative
#(1/(1+np.exp(-dummyInput)))*(1-(1/(1+np.exp(-dummyInput))))
Oin = layersResultBefore.pop()
dOoutbydOin = (1/(1+np.exp(-Oin)))*(1-(1/(1+np.exp(-Oin))))

prevOut = layersResultAfter.pop()

#number of columns represent the number of outputs
#number of rows represent input
dWbydA = np.zeros((prevOut.shape[1], Oout.shape[1]))
print(prevOut.shape, dWbydA.shape)
for i in range(Oout.shape[1]):
    dWbydA[:, i] = prevOut
# layersResultAfter.append(prevOust)
dEdW = dWbydA*dOoutbydOin*dEbydOout
dEdB = dOoutbydOin*dEbydOout
# print(dEdW.shape, initialWeights[0][-1].shape)
print(initialWeights[1][-1])
#weights update
weight = initialWeights[0][-1]
bias = initialWeights[1][-1]
weight = weight - lr*dEdW
bias = bias - lr*dEdB
updatedWeights = []
updatedWeights.append([weight, bias])
print(initialWeights[1][-1])
Hin = layersResultBefore.pop()
print(Hin.shape)
dHoutdHin = np.where(Hin>0, Hin, 0)
dHindW = layersResultAfter.pop()
#dEtotaldA = dEdOout*dOoutdOin*dOindA
dEtotaldA = dEbydOout*dOoutbydOin*initialWeights[0][-1]
print(dEtotaldA.shape, dHoutdHin.shape, dHindW.shape)
dEdW1 = np.linalg.multi_dot([np.transpose(dHindW),dHoutdHin,dEtotaldA])
dEdB1 = np.dot(dHoutdHin, dEtotaldA)

weight = initialWeights[0][-2]
bias = initialWeights[1][-2]
weight = weight - lr*dEdW1
bias = bias - lr*dEdB1
updatedWeights.append([weight, bias])
print(bias.shape, weight.shape)






        


    

