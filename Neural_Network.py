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

import time
from threading import Thread

#Neural Network Class
class Neural_Network:
    def __init__(self):
        self.runner = Runner()
        self.input = 4
        self.middle = 4
        self.output = 3
        #In the output the player can go up or down
        self.weights_1 = np.random.randn(self.middle,self.input)
        self.weights_2 = np.random.randn(self.middle,self.output)
        print(self.weights_1)
        print(self.weights_2)
        
    def forward(self, x):
        #input_x = copy.deepcopy(x)
        self.input_middle = np.dot(self.weights_1,x)
        self.input_middle = self.sigmoid(self.input_middle)
        self.middle_output = np.dot(self.input_middle,self.weights_2)
        self.output_layer = self.sigmoid(self.middle_output)
        return self.output_layer
    
    def sigmoid(self, value):
        return 1/(1+np.exp(-value))

    def sigmoidPrime(self, value):
        return value*(1-value)
    
    def backward(self,given_input,expected_output,predicted_output):
        #output_calculated = np.array([predicted_output[0] - predicted_output[2],predicted_output[1] - predicted_output[3]])
        self.error = expected_output - predicted_output
        self.delta = self.error * self.sigmoidPrime(predicted_output)
        self.output_error = self.delta.dot(self.weights_2.T)
        self.d_delta = self.output_error * self.sigmoidPrime(self.input_middle)
        self.weights_1 = given_input.T.dot(self.d_delta)
        self.scaled_middle = np.array([self.input_middle[0],self.input_middle[1],self.input_middle[2]])
        #self.weights_2 = self.scaled_middle.T.dot(self.delta)
        print("Weights_1",end=": ")
        print(self.weights_1)

    def train(self, X, y):
        #Save Code to show Andrew for his project
        game_thread = Thread(target = self.runner.game_start)
        game_thread.setDaemon(True)
        game_thread.start()
        #p = multiprocessing.Process(target=self.runner.game_start)
        #p.start()
        #self.predict()
        while not self.runner.intro:
            print("wait for the game to start")
            self.runner.intro = False
            break
            time.sleep(1)
        input_x = np.array([self.runner.player_pos_x,self.runner.player_pos_y,self.runner.obst.x,self.runner.obst.y])
        while self.runner.exitGame:
            time.sleep(.5)
        while not self.runner.exitGame:
            #time.sleep(.5)
            if(self.runner.exitGame):
                break
            input_x = np.array([self.runner.player_pos_x,self.runner.player_pos_y,self.runner.obst.x,self.runner.obst.y])
            player_x = self.runner.player_pos_x
            obstacle_x = self.runner.obst.x
            player_y = self.runner.player_pos_y
            obstacle_y = obstacle_x = self.runner.obst.x
            #Check if the Array is a 0D Array or = None
            if input_x.all() == None:
                break
            #Forward Propagation := Making the decision to move up, down or stay the same
            output = self.forward(input_x)
            output_y = np.array([self.runner.player_pos_x,self.runner.player_pos_y,self.runner.obst.x,self.runner.obst.y])
            print("Output: ",)
            print(output)
            maxed = max(output)
            if maxed == output[0]:
                print("Option A")
                self.runner.move_up()
                output_y = np.array([self.runner.player_pos_x,self.runner.player_pos_y,self.runner.obst.x,self.runner.obst.y])
            elif maxed == output[1]:
                print("Option B")
                time.sleep(1)
            elif maxed == output[2]:
                print("Option C")
                self.runner.move_down()
                output_y = np.array([self.runner.player_pos_x,self.runner.player_pos_y,self.runner.obst.x,self.runner.obst.y])
            else:
                print("Default Option")
                time.sleep(1)
                continue
            #What should our Y be in order for this to work out perfectly??
            y = [0,0,0]
            if obstacle_y == player_y:
                y = [1,0,1]
            else:
                y = [0,1,0]
            self.backward(input_x,y,output)
        self.saveWeights()
        #net_thread = Thread(target = self.predict)
        #net_thread.setDaemon(True)
        #net_thread.start()
        

    def saveWeights(self):
        file_weights1 = open("weights_1.txt","w")
        file_weights1.write(str(self.weights_1))
        #np.savetxt("weights_1.txt",self.weights_1,fmt="%s")
        np.savetxt("weights_2.txt",self.weights_2,fmt="%s")
    def lossFunction(self,predicted_y,actual_y):
        #We will use Mean Squared Error for our loss
        # Loss = sum of (pred_y - actual_y)^2
        error = (predicted_y - actual_y)**2 # **2 is squaring it
        error = np.sum(error)
        error = error/actual_y.size
        return error
    
    def predict(self):
        print("")
