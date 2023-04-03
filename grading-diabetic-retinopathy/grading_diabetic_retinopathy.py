from keras import layers
from keras import models
from keras import utils
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os, shutil
import cv2

image_size = (512, 512)

#Create a Pandas dataframe for training data for exploratory data analysis
df = pd.read_csv(r'../images/trainLabels.csv')

#Total rows of data in our dataframe (-1 due to first row consisting of column headers)
print("Columns: ", len(df.columns), "- Rows: ", len(df.index))

#Check for any duplicated rows or null values
print(f"There are {df.duplicated().sum()} repeated rows")
print(df.isnull().sum())

#Visualisation of quantity of images related to their grade of retinopathy
distribution = sns.countplot(x=df['level'])
plt.show()

#

#Assign training/testing to np.arrays
def train_load(dir):
    temp = []
    i = 0
    for img in os.listdir(dir):
        try:
            arr = cv2.imread(os.path.join(dir, img), cv2.IMREAD_GRAYSCALE)
            resize = cv2.resize(arr, (image_size))
            temp.append([resize, df.loc[i, 'level']])
        except Exception as exception:
            print(exception)
    return np.array(temp)

train = train_load(r'../images/train')

print(train.shape())









