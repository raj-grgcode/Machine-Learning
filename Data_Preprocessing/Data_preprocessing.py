import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('train.csv')
print(df.shape)
print(df.info())
print(df.describe())
print(df.head())
print(df.isnull().sum())

#Age
#So after running the code we can see age has 177 data missing which is about 20%
#so lets fill it with median
df['Age'].fillna(df['Age'].median(), inplace=True)
#Cabin
#Cabin column has 687 data missing out of 891 data which is lot so lets drop it
df.drop(columns=['Cabin'],inplace=True)
print(df.isnull().sum())