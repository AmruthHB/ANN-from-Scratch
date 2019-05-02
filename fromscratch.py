# import numpy
import numpy as np

# imported
x = np.array(([[78.5 ,67],
               [75,64],
               [75,58.5] ]), dtype = float)

y = np.array(([73.2],
             [71],
             [70.5]), dtype = float)

#standardize
x = x/np.amax(x,axis = 0)
y = y/np.amax(y,axis = 0)
#max test score is 100

class Neural_Network():
    def __init__(self):
        #hyper params
        self.inputLayerSize = 2
        self.hiddenLayerSize = 3
        self.outputLayerSize = 1
        
        
        #weights
        
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize) #3x2
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    
    def forward(self,x):
        self.z2 = np.dot(x,self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2,self.W2)
        self.yHat = self.sigmoid(self.z3)
        
        return self.yHat
    
    def Sigmoid_Prime(self,x):
        return np.exp(-x)/(1+np.exp(-x))**2
    
    def Cost_Function(self,x,y):
        #cost function: j = sigma 1/2(y-yhat)**2
        self.yHat = self.forward(x)
        J = 0.5*sum((y-self.yHat))
        return J
    
    def Cost_Function_Prime(self,x,y):
        self.yHat = self.forward(x)
        
        delta_3 = np.multiply(-(y-self.yHat),self.Sigmoid_Prime(self.z3))
        dJdW2 =  np.dot(a.T,delta3)
        
        #dJdW1
        
        delta_2 = np.dot(delta_3,W2.T)*self.Sigmoid_Prime(self.z2) #hadamard product
        # constant of backpropogation for this layer
        
        dJdW1 = np.dot(x.T, delta_2)
        
        return dJdW1, dJdW2
    
    
    
    
#network = Neural_Network()
#print(network.forward(x))
