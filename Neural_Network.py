'''
Creating a Neural Network that can play a simple runner game
'''
from Runner import *
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy
import math
import sklearn
import sklearn.datasets
import sklearn.linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
np.random.seed(0)

import time
from threading import Thread

#Neural Network Class
class Neural_Network:
    def __init__(self):
        self.runner = Runner()
        self.input = 3
        self.middle = 6
        self.output = 3
        self.learning_rate = 0
        #In the output the player can go up or down
        self.weights_1 = np.random.randn(self.middle,1)
        self.weights_2 = np.random.randn(self.output,1)
        weight_fileA = open("weights_1.txt","r")
        weight_fileB = open("weights_2.txt","r")
        print("Weights 1: ",end="")
        print(self.weights_1)
        print("Weights 2: ",end="")
        print(self.weights_2)

    def dot_product(self,array,multiplier):
        product = 0
        print("DotProduct Start**************************************************************")
        for i in range(len(array)):
            product+= array[i]*multiplier
        print(product)
        print("Dot Product End**************************************************************")
        return product
            
    def forward_layer(self,layer1, weights):
        result_layer = []
        for i in range(len(weights)):
            result_layer.append(self.dot_product(layer1,weights[i]))
        return np.array(result_layer)
    def forward(self, x):
        self.input_middle = self.forward_layer(x,self.weights_1)# np.dot(self.weights_1,x)
        self.input_middle = self.sigmoid(self.input_middle)
        print(self.input_middle)
        self.middle_output = self.forward_layer(self.input_middle,self.weights_2)#np.dot(self.input_middle,self.weights_2)
        print(self.middle_output)
        self.output_layer = self.sigmoid(self.middle_output)
        output = [self.output_layer[0],self.output_layer[1],self.output_layer[2]]
        return output
    
    def sigmoid(self, value):
        return 1/(1+np.exp(-value))

    def sigmoidPrime(self, value):
        return value*(1-value)

    def calculate_error(self,predicted_output,actual_output):
        error = []
        for i in range(len(actual_output)):
            error.append((1/2)*(math.pow((predicted_output[i]-actual_output[i]),2)))
        error = np.array(error)
        return error
    
    def subtract_arr(self,arr1,arr2):
        print("subtract_arr Start*********************************************************")#There is no error checking for the size of the array
        result = []
        for i in range(len(arr1)):
            result.append(arr1[i] - arr2[i])
        print(result)
        print("subtract_arr END*********************************************************")
        return result
    def multiply_arr(self, array1, array2):
        if len(array1) != len(array2):
            return
        result = []
        for i in range(len(array1)):
            result.append((array1[i]*array2[i]))
        return result
    def combine_arr(self,array1,array2):
        combined = []
        for i in range((len(array1)+len(array2))):
            if i < len(array1):
                combined.append(array1[i])
            if i >= len(array1):
                combined.append(array2[i-len(array1)])
        return combined
    def backward(self,given_input,expected_output,predicted_output):
        print("Backward Start*********************************************************")
        self.delta = self.subtract_arr(predicted_output,expected_output)
        copy_weights = self.weights_2
        #Get the dot product of delta times learning rate
        dotproduct_delta = self.dot_product(self.delta,self.learning_rate)
        self.weights_2 = self.subtract_arr(self.weights_2,dotproduct_delta*(self.weights_2))
        #self.weights_1 = self.weights_1 - dotproduct_delta*(given_input)*copy_weights
        weight1_update = dotproduct_delta*(self.multiply_arr(given_input,copy_weights))
        split_weight1a = [self.weights_1[0:3]]
        split_weight1b = [self.weights_1[3:]]
        self.weights_1 = self.combine_arr((self.subtract_arr(split_weight1a,weight1_update)),(self.subtract_arr(split_weight1b,weight1_update)))
        self.weights_1 = np.concatenate(self.weights_1)
        print("Weights 1: ",end="")
        print(self.weights_1)
        print("Weights 2: ",end="")
        print(self.weights_2)
        print("Backward End*********************************************************")

    def train(self, X, target,learning_rate):
        #Set learning rate to new learning rate
        self.learning_rate = learning_rate
        #Run the game on different thread so nothing freezes
        game_thread = Thread(target = self.runner.game_start)
        game_thread.setDaemon(True)
        game_thread.start()
        #This while loop is to skip the intro
        while not self.runner.intro:
            print("wait for the game to start")
            self.runner.intro = False
            break
            time.sleep(1)
            input_x = np.array([self.runner.player_pos_x,self.runner.player_pos_y,self.runner.obst.y])#,self.runner.obst.y])
        #Wait while the game is loaded
        while self.runner.exitGame:
            time.sleep(.5)
        #Now the game is loaded so we can use the neural network
        while not self.runner.exitGame:
            if(self.runner.exitGame):
                break
            input_x = np.array([self.runner.player_pos_x,self.runner.player_pos_y,self.runner.obst.y])
            player_x = self.runner.player_pos_x
            obstacle_x = self.runner.obst.x
            player_y = self.runner.player_pos_y
            obstacle_y = obstacle_x = self.runner.obst.x
            #Check if the Array is a 0D Array or = None
            if input_x.all() == None:
                break
            option = ""
            #Forward Propagation := Making the decision to move up, down or stay the same
            output = self.forward(input_x)
            output_y = [self.runner.player_pos_x,self.runner.player_pos_y,self.runner.obst.x,self.runner.obst.y]
            print("Output: ",)
            print(output)
            maxed = max(output)
            if maxed == output[0]:
                print("Option A")
                option = "A"
                self.runner.move_up()
                output_y = [self.runner.player_pos_x,self.runner.player_pos_y,self.runner.obst.y]#,self.runner.obst.y])
                time.sleep(1)
            elif maxed == output[1]:
                print("Option B")
                option = "B"
                self.runner.move_down()
                time.sleep(1)
            elif maxed == output[2]:
                print("Option C")
                option = "C"
                self.runner.move_down()
                output_y = [self.runner.player_pos_x,self.runner.player_pos_y,self.runner.obst.y]
                time.sleep(1)
            else:
                print("Default Option")
                time.sleep(1)
                continue
            #What should our Y be in order for this to work out perfectly??
            y = [0,0,0]
            if obstacle_y == player_y:
                y = [0,0,1]
            elif obstacle_y + self.runner.obst.radius > player_y:
                y = [1,0,0]
            elif obstacle_y - self.runner.obst.radius <= player_y:
                y = [0,0,1]
            else:
                y = [0,1,0]
            if self.runner.obst.y - self.runner.obst.radius <= player_y:
                y = [0,0,1]
            if self.runner.obst.y + self.runner.obst.radius >= player_y:
                y = [1,0,0]
            if player_y +50 >= self.runner.window_height:
                y = [1,0,0]
            if player_y - 50 <= 0:
                y = [0,0,1]
            if self.runner.crashed:
                if option == "A":
                    y = [0,0,1]
                if option == "B":
                    y = [1,0,1]
                if option == "C":
                    y = [1,0,0]
                #game_thread._Thread_stop()
                if self.runner.score >= target:
                    self.runner.quitGame = True
            time.sleep(2)
                    
            #Backward Propogation to make the neural network learn
            self.backward(input_x,y,output)
            print(self.runner.score)
        self.saveWeights()
        

    def saveWeights(self):
        file_weights1 = open("weights_1.txt","w")
        file_weights1.write(str(self.weights_1))
        file_weights2 = open("weights_2.txt","w")
        file_weights2.write(str(self.weights_2))
        
    def lossFunction(self,predicted_y,actual_y):
        #We will use Mean Squared Error for our loss
        # Loss = sum of (pred_y - actual_y)^2
        error = (predicted_y - actual_y)**2 # **2 is squaring it
        error = np.sum(error)
        error = error/actual_y.size
        return error
    
    def predict(self):
        print("")

nn = Neural_Network()
nn.forward(np.array([50,20,100]))
