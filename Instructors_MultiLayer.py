# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 20:33:59 2024

@author: Modified Instructors Code
"""

import numpy as np
import random


class perceptron(object):
    
    def __init__(self, init_weights, bias):
        # init_weights is initial weight vector
        # self.OutFlag = OutFlag
        self.weights = np.array(init_weights,dtype=float) # initialize weight vector
        self.delta_weights = np.array(init_weights,dtype=float) # init delta weight vector
        self.bias = bias # init bias value
        self.delta_bias = 0.
        self.delta = 0 # init delta value
        self.vector_length = len(self.weights)
        self.activity = 0 # init activity value
        self.activation = 0 # init activation value
        self.index = 0 # index value of perceptron in a layer
        self.error = 0 # only really useful for output nodes
    
    def calc_activity(self, input_vector):
        #input = np.array(input_vector,dtype=float) # make this a numpy array
        A = 0 #init activity value 
        A = np.dot(input_vector,self.weights)
        # for i in range(0,self.vector_length): # calc each term
        # A += input[i] * self.weights[i]
        A += self.bias # add bias value
        self.activity = A
    
    def calc_activation(self, input_vector):
        self.calc_activity(input_vector)
        self.activation = 1/(1.0 + np.exp(-1*self.activity))
        if self.activation >= 0.8: # perceptron threshold logic
            self.activation = 1
        else:
            self.activation = 0

    def get_activation(self):
        # self.calc_activation(input_vector)
        return(self.activation)
       
    def set_delta(self,delta):
        self.delta = delta
    
    def get_delta(self):
        return(self.delta)
    
    def set_index(self,index):
        self.index = index
    
    def get_index(self):
        return(self.index)
    
    def get_weight_element(self,weight_vector_index):
        weight = self.weights[weight_vector_index]
        return(weight)
    
    def set_delta_weights(self,eta,input_vector):
        for i in range(0, self.vector_length):
            xi = input_vector[i]
            self.delta_weights[i] = eta*self.delta*xi
    
    def update_weights(self):
        self.weights = self.weights + self.delta_weights
        self.bias = self.bias + self.delta_bias
        
class NodeLayer(object):
    
    def __init__(self, OutputLayerFlag, num_inputs, num_nodes, init_weights, bias):
        # first set up loop and append perceptrons to list
        self.layer = [] #create empty layer list
        self.layer_length = num_nodes
        self.numInputs = num_inputs
        self.OutputFlag = OutputLayerFlag #true if output layer
        self.Layer_weights = init_weights
        # if output layer, setup littleE_vector
        if OutputLayerFlag == True:
            self.littleE_vector = np.zeros(num_nodes,dtype=float)
        # init output vector array
        self.output_vector = np.zeros(num_nodes,dtype=float)
        # self.input_vector = np.array(num_nodes)
        #determine if input layer, hidden or output layer
        #create a list of perceptrons of length num_nodes
        for i in range (0, num_nodes):
            #determine perceptron weight vectors by slicing
            j = i*num_inputs #index for the node
            perceptron_weights = init_weights[j:j+num_inputs]
            self.layer.append(perceptron(perceptron_weights,bias))
    
    def get_err_vector(self,desired):
        if self.OutputFlag == True:
            self.littleE_vector = desired - self.output_vector
            #for i in range (0,self.layer_length):
            # self.layer[i].error = desired[i] - self.output_vector[i]
        return (self.littleE_vector)
    
    def get_layer_output_vector(self,input_vector):
        for i in range(0,self.layer_length):
            self.layer[i].calc_activation(input_vector)
            self.output_vector[i] = self.layer[i].get_activation()
        return(self.output_vector)
    
    def get_layer_length(self):
        return(self.layer_length)
    
    def set_output_layer_delta_values(self,desired):
        if self.OutputFlag == False:
            print("You messed up. This is not an output layer!")
            exit()
        else: 
            self.litteE_vector = self.get_err_vector(desired)
            for i in range(0,self.layer_length):
                yk = self.output_vector[i]
                deltak = self.littleE_vector[i]*yk*(1-yk)
                self.layer[i].set_delta(deltak)
#                print("Output layer delta = " + str(deltak))
                
    def set_hidden_layer_weighted_delta_values(self,abovelayer):
        # above layer is a layer list for the above layer
        above_length = abovelayer.get_layer_length()
        delta = 0 #initialize delta value
        if self.OutputFlag == False:
            #do stuff to compute delta values for hidden layer
            for i in range (0, self.layer_length): # i is current layer index
                delta = 0. # init delta for node i
                weighted_sum_delta = 0.
                #get output value corresponding to node i
                xi = self.output_vector[i]
                for j in range(0,above_length):
                    #consider al nodes in layer above
                    above_delta = abovelayer.layer[j].get_delta()
                    link_weight = abovelayer.layer[j].get_weight_element(i)
                    weighted_sum_delta += above_delta*link_weight
                delta = weighted_sum_delta*xi*(1-xi)
                self.layer[i].set_delta(delta)
    
    def calc_layer_delta_weights(self,eta,input_vector):
        for k in range(0,self.layer_length): # for each output layer node
            self.layer[k].set_delta_weights(eta,input_vector)
    
    def update_layer_weights(self):
        for k in range(0,self.layer_length): # for each output layer node
            self.layer[k].update_weights()

#define input/output pairs as numpy arrays
# ...parameter setup
input = np.array([[0.9,0.87],
            	 [1.31,0.75],
            	 [2.48,1.14],
            	 [0.41,1.87],
            	 [2.45,0.52],
            	 [2.54,2.97],
            	 [0.07,0.09],
            	 [1.32,1.96],
            	 [0.94,0.34],
            	 [1.75,2.21]],
                 float)
desiredOut = np.array([[1],[1],[0],[0],[0],[1],[1],[0],[1],[0]],float)
eta = 1.0

weights_hidden = [random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)]
weights_output = [random.uniform(0,1),random.uniform(0,1)]
bias = random.uniform(0,1)

#network architecture setup

hiddenlayer = NodeLayer(False,2,2,weights_hidden,bias)
outputlayer = NodeLayer(True,2,1,weights_output,bias)

for iter in range(0,15):
    #feedforward epoch
    for pair in range(0,10):
        hidden_out = hiddenlayer.get_layer_output_vector(input[pair,:])
        output_out = outputlayer.get_layer_output_vector(hidden_out)
        Error = outputlayer.get_err_vector(desiredOut[pair,:])
   
        #initiate backpropogation
        
        outputlayer.set_output_layer_delta_values(desiredOut[pair,:])
        hiddenlayer.set_hidden_layer_weighted_delta_values(outputlayer)
        
        #now calc delta weights for the layers
        
        outputlayer.calc_layer_delta_weights(eta,hidden_out)
        hiddenlayer.calc_layer_delta_weights(eta,input[pair,:])
        
        #now update the weights
        
        outputlayer.update_layer_weights()
        hiddenlayer.update_layer_weights()
        Error = outputlayer.get_err_vector(desiredOut[pair,:])
        BigE = 0.5*Error**2
#        print("The activation function value of Nodes 1 and 2 is: ", hidden_out)
#        print("The activation function value of Node 3 is : ", output_out)
#        print("Input " + str(input[pair,:]))
#        print("Iteration " + str(iter) + " Desired Value " + str(desiredOut[pair,:]) + " Actual Value " + str(BigE))
#        print("Big E for iteration " + str(iter) + "=" + str(BigE))

#end loops -- print final results
input_val = np.array([[1.81,1.02],
            	     [2.36,1.6],
            	     [2.17,2.08],
            	     [2.85,2.91],
            	     [1.05,1.93],
            	     [2.32,1.73],
            	     [1.86,1.31],
            	     [1.45,2.19],
            	     [0.28,0.71],
            	     [2.49,1.52]],
                     float)

desiredOut = np.array([[0],[0],[1],[1],[0],[0],[0],[0],[1],[0]],float)

for pair in range(0,10):
    hidden_out = hiddenlayer.get_layer_output_vector(input_val[pair,:])
    output_out = outputlayer.get_layer_output_vector(hidden_out)
    Error = outputlayer.get_err_vector(desiredOut[pair,:])
    BigE = 0.5*Error**2
    print("Input " + str(input_val[pair,:]))
    print("Desired Value " + str(desiredOut[pair,:]) + " Actual Value " + str(BigE))

