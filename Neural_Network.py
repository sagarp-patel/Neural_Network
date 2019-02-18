'''
Creating a Neural Network that can play a simple runner game
'''
from Runner import *
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import sklearn
import sklearn.datasets
import sklearn.linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
np.random.seed(0)

#Neural Network Class
class Neural_Network:
    def __init__(self):
        self.runner = Runner()
        self.input = 3
        self.middle = 4
        self.output = 3
        #In the output the player can go up or down
        self.weights_1 = np.random.randn(self.middle,1)
        self.weights_2 = np.random.randn(self.output,1)
        print("This is a Neural Network")
        
    def forward(self, x):
        self.input_middle = np.dot(x,self.weights_1)
        self.input_middle = self.sigmoid(self.input_middle)
        self.middle_output = np.dot(self.input_middle,self.weights_2)
        self.output_layer = self.sigmoid(self.middle_output)
        return self.output_layer
    
    def sigmoid(self, value):
        return 1/(1+np.exp(-value))

    def sigmoidPrime(self, value):
        return value*(1-value)
    def backward(self,X,y,o):
        print("nothing to return here")

    def train(self, X, y):
        self.runner.game_start()
        input_x = [self.player_pos_x,self.player_pos_y,,]
        for i in range(10):
            input_x = [self.runner.]
            self.forward()
            
        print("nothing to return here")

    def saveWeights(self):
        print("nothing to return here")
    def lossFunction(self,predicted_y,actual_y):
        #We will use Mean Squared Error for our loss
        # Loss = sum of (pred_y - actual_y)^2
        error = (predicted_y - actual_y)**2 # **2 is squaring it
        error = np.sum(error)
        error = error/actual_y.size
        return error
    
    def predict(self):
        print("Hello")
