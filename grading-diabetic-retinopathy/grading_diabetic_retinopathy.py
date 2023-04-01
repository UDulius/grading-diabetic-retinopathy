from keras import layers
from keras import models
from keras import utils
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os, shutil

#Assign training and testing images for exploratory data analysis

#Create a Pandas dataframe for training data for exploratory data analysis
df = pd.read_csv(r'../images/trainLabels.csv')

#Total rows of data in our dataframe (-1 due to first row consisting of column headers)
print("Columns: ", len(df.columns), "- Rows: ", len(df.index))

#Check for any duplicated rows or null values
print(f"There are {df.duplicated().sum()} repeated rows")
print(df.isnull().sum())

#Visualisation of quantity of images related to their grade of retinopathy
#distribution = sns.countplot(x=df['level'])
#plt.show()

Test 




