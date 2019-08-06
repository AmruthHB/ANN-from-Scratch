#References
#https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

import numpy as np
import pandas as pd
import math

#get dataset from files
housing_prices = pd.read_csv("Taiwan.csv")

#drop first dow
housing_data = housing_prices.iloc[:,1:]

#numpy matrix of data
feed_data = housing_data.values

#split into x and y dataset
X = feed_data[:,0:-1].astype('float32')
y = feed_data[:,-1].astype('float32')


#keep a copy of y without scaling to remove feature scaling in the future
y_unscaled = np.copy(y)



#for each col take max value in col and feature scale
#https://medium.com/greyatom/why-how-and-when-to-scale-your-features-4b30ab09db5e
X = (X-np.amin(X,axis = 0))/(np.amax(X, axis=0)-np.amin(X,axis = 0))

#likewise for y
y = (y-np.amin(y,axis = 0))/(np.amax(y, axis=0)-np.amin(y,axis = 0))

#*to rescale rearrange equation  x = (x' * (np.amax-np.amin))+np.amin


#split into train and text sets at 70 30 split
X_train = X[0:int(round(X.shape[0]*0.7,0)),:]
X_test = X[int(round(X.shape[0]*0.7,0))+1:,:]

y_train = y[0:int(round(y.shape[0]*0.7,0))].reshape((290,1))
y_test = y[int(round(X.shape[0]*0.7,0))+1:].reshape((123,1))

#Neural Net Class
class Neural_Net():
    
    def __init__(self):
        
        #layer hyper param
        self.inputs = 6
        self.layer_1 = 4
        self.output = 1
        
        #learning rate
        self.learning_rate = 0.07
        
        #weights
        self.W1 = np.zeros((self.inputs,self.layer_1))
        self.W2 = np.zeros((self.layer_1,self.output))
        
        #biases
        self.b1 = np.zeros((1,self.layer_1))
        self.b2 = np.zeros((1,self.output))
        
        
    def weights_initialize(self,m,n):
        
       #https://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
       #https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/
      
       #caffe library does it ifferently for some reason 
       n_avg = (m+n)/2
       var_wi = 1/n_avg
       
       W = np.random.normal(loc=0.0, scale= math.sqrt(var_wi), size=[m,n])
       
       return W
       
    #bias same dim as second layer values
    #https://stackoverflow.com/questions/44883861/initial-bias-values-for-a-neural-network
    def bias_initialize(self,m,n):
        
        bias = [0.0001]*(m*n)
        bias = np.array(bias).reshape((m,n))
        
        return bias   
        
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def sigmoid_prime(self,z):
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def forward_propogation(self,x):
    
        self.z2 = np.add(np.dot(x,self.W1),self.b1)
        self.a2 = self.sigmoid(self.z2)
        
        self.z3 = np.add(np.dot(self.a2,self.W2),self.b2)
        
        y_hat = self.sigmoid(self.z3)
        
        return y_hat
    
    def cost_function(self,y,y_pred):
        
        #self.y_hat = self.forward_propogation(X)
        C = (1/(2*29))*((y-y_pred)**2)
        
        return C
    
    def cost_function_derivative(self,X,y,y_pred):
        
        #calculated from simplyfying terms when manually calculating partial derivatives
        sigma_2 = np.multiply(np.multiply(-1/3,(y-y_pred)),self.sigmoid_prime(self.z3))
        
        #derivate of W2 with respect to Cost
        dCdW2 = np.dot(self.a2.T,sigma_2)
        
        #calculated from simplyfying terms when manually calculating partial derivatives
        sigma_1 = np.multiply(np.dot(sigma_2,self.W2.T),self.sigmoid_prime(self.z2))
        
        #derivate of W2 with respect to Cost
        dCdW1 = np.dot(X.T,sigma_1)
        
        return dCdW1,dCdW2,
        
    
    def update_weights(self,derivatives):
        
        #copy paramans into new array
        weight_derivative = derivatives
        
        #https://math.stackexchange.com/questions/1972640/matrix-multiplication-question-of-2-3x1-vectors 2 3x1 dots multiplications
        self.W1 = self.W1 - self.learning_rate*weight_derivative[0]
        self.W2 = self.W2 - self.learning_rate*weight_derivative[1]
     
    #implementation of Root Mean Square error to see how well predictions compare to actual data
    def RMSE(self,y,y_pred):
        RMSE_val = np.sqrt(np.sum((y_pred-y)**2)/y.shape[0])
        
        return RMSE_val
        

NN = Neural_Net()

#initizlize biases and weights
NN.W1 = NN.weights_initialize(6,4)
NN.W2 = NN.weights_initialize(4,1)

NN.b1 = NN.bias_initialize(1,NN.layer_1)
NN.b2 = NN.bias_initialize(1,NN.output)


#set epochs
epochs = 3
for i in range(epochs):
    
    #feed in mini batches of 29 training examples
    start_idx = 0
    stop_idx = 29
    print("epoch",i+1,"losses: \n")
    
    #10 sets of 29 batches fed in
    for m in range(int(X_train.shape[0]/29)):
        
        #keep feeding while indexing variables have array index in range
        if stop_idx <= int(X_train.shape[0]):
            mini_batch = X_train[start_idx:stop_idx,:]
            
            #make predictions from batch
            y_pred = NN.forward_propogation(X_train[start_idx:stop_idx,:])
            
           #calcluate cost function
            loss = NN.cost_function(y_train[start_idx:stop_idx],y_pred)
            display_loss = np.sum(loss,0)
            print("batch",m+1, "loss: ",float(display_loss))
            
            #get derivatives of weights with repoect to cost function
            dW1,dW2 = NN.cost_function_derivative(mini_batch ,y_train[start_idx:stop_idx],y_pred)
        
            #update weights of neural network
            NN.update_weights([dW1,dW2])
        
            #move  to next batch
            start_idx += 29
            stop_idx += 29
            
    print("\n \n")
           
#predict with adjusted weights
test_pred = NN.forward_propogation(X_test)

#convert values back for RMSE + comparision
de_scaled_y = (y_test*(np.amax(y_unscaled, axis=0)-np.amin(y_unscaled,axis = 0)))+np.amin(y_unscaled,axis = 0)
test_pred_unscaled = (test_pred*(np.amax(y_unscaled, axis=0)-np.amin(y_unscaled,axis = 0)))+np.amin(y_unscaled,axis = 0)

#TEST ACCURACY
final_model_accuracy = NN.RMSE(y_test,test_pred)

#RMSE OF 0.15 achived (not bad)












