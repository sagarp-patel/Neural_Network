'''
Following a Tutorial from freecodecamp.com to learn how Neural Networks work and
how to implement one.
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
_problem_to_solve = "A farmer in Italy was having a problem with his labelling machine: it mixed up the labels of three wine cultivars. Now he has 178 bottles left, and nobody knows which cultivar made them! To help this poor man, we will build a classifier that recognizes the wine based on 13 attributes of the wine."

np.random.seed(0)

class Neural_Network:
    def __init__(self):
        self.runner = Runner()
        print("This is a Neural Network")
        
    def forward(self, x):
        o = 10
        return o
    
    def sigmoid(self, s):
        return s

    def sigmoidPrime(self, s):
        return s
    def backward(self,X,y,o):
        print("nothing to return here")

    def train(self, X, y):
        self.runner.game_start()
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
