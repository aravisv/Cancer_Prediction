# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 22:11:37 2019

@author: aravi
"""

import numpy as np
import pandas as pd

data=pd.read_csv('breastcancer.csv')
print(data)

X=data.iloc[:,:9] # 9 features
print(X)

y=data.iloc[:,9]    #output
print(y)

print(np.shape(X))  #matrix of order 116 X 9

print(np.shape(y))  #array 

# so we need y to convert it to matrix using the command below

y=y[:,np.newaxis]
print(y)
print(np.shape(y))

#we can see that y has become a column vector or matrix of order 116 x 1

#mean normalization
X=(X-np.mean(X))/np.std(X)


#activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#find the slope or gradient 
def slope(s):
    return s*(1-s)


epochs=5000 #number of iterations
alpha=0.1   #learning rate
ip_neurons=X.shape[1]   #shape[1] returns number of columns
hidden_layers=16    #number of hidden layer neurons
output=1    #number of layers in output


#variable initialization
weights=np.random.uniform(size=(ip_neurons,hidden_layers))  #matrix of order ip_neurons x hidden_layers
weights_out=np.random.uniform(size=(hidden_layers,output))  #matrix of order hidden_layers x output
bias=np.random.uniform(size=(1,hidden_layers))  #matrix of order 1 x hidden_layers
bias_out=np.random.uniform(size=(1,output)) #matrix of order 1 x output

#print(weights,weights_out,bias,bias_out)

for i in range(epochs):
    #forward propogation
    hiddenlayer_ip=np.dot(X,weights)
    hiddenlayer_ip=hiddenlayer_ip+bias
    activation1=sigmoid(hiddenlayer_ip)
    ipfor_op=np.dot(activation1,weights_out)
    ipfor_op=ipfor_op+bias_out
    output=sigmoid(ipfor_op) 

    #backward propogation
    error=y-output
    slope_op_layer=slope(output)
    slope_hiddenlayer=slope(activation1)
    delta_op=error*slope_op_layer
    error_hidden=np.dot(delta_op,weights_out.T)
    delta_hidden=error_hidden*slope_hiddenlayer
    weights_out+=np.dot(activation1.T,delta_op) *alpha
    bias_out+=np.sum(delta_op,axis=0)*alpha
    weights+=np.dot(X.T,delta_hidden)*alpha
    bias+=np.sum(delta_hidden,axis=0)*alpha

print(weights)
print(output)