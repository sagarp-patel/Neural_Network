'''
Following a Tutorial from freecodecamp.com to learn how Neural Networks work and
how to implement one.
'''
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
