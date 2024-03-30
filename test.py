# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 20:57:39 2023

@author: okokp
"""


import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
# We will need these to convert our categorical label into 1's and 0's and our entire image to a matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array,load_img

# We will use this to encode our categorical label
from sklearn.preprocessing import LabelEncoder 
root_dir = "C:/Users/Sahana Gowda/Desktop/Object_intr/train/"
df_train = pd.read_csv('Training_set.csv')
df_test = pd.read_csv('Testing_set.csv')
     

df_train.head()
df_test.head()

# We don't have a label on our testing data
     


df_train.describe()

# As expected, we have 12600 rows
# 15 unique labels to represent each of our actions
# A frequency of 840 implies that all labels have an equal amount of rows containing them

actions = df_train['label'].unique()
for i in range(len(actions)):
  print('Object Interaction', i + 1 , ":", actions[i].capitalize().replace("_", " "))
  
# We want to split our training data so that the features and the labels are in seperate matrices

img_data = []
img_label = []

for i in range(len(df_train)):
    img = root_dir + df_train['filename'][i]
    img = load_img(img, target_size = (150,150)) # By convention, pictures are either 96x96 or 256x256. 150x150 strikes a nice balance between the two
    img = img_to_array(img)
    img_data.append(img)
    img_label.append(df_train["label"][i])
     

# We need to work on a matrix, so we convert our arrays using np.array()

img_data = np.array(img_data)
img_label = np.array(img_label)
     

img_data.shape

# Our data matrix has a dimension of 3, and contains the number of rows, pixels on the X axis, and pixels on the Y axis

img_label.shape

# Our label matrix is just a column vector containing the labels for our data matrix
     
# Create an instance of the class
encoder = LabelEncoder()

#Fit our data 
img_label = encoder.fit_transform(img_label)
     

img_label = to_categorical(img_label)
print(img_label)
     

import random
from matplotlib import image as img

def display_random(n):
    
    plt.figure(figsize=(15, 20))
    for i in range(n):
        rnd = random.randint(0, len(df_train)-1)
        img_file = root_dir + df_train['filename'][rnd]

        
        plt.subplot(n//2+1, 2, i + 1)
        image = img.imread(img_file)
        image = load_img(img_file,target_size = (150,150))
        plt.imshow(image)
        plt.title(df_train['label'][rnd])
     

display_random(5)

