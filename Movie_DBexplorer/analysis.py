import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize']=(12,6)

# Now the CSV is in the Data subfolder
df = pd.read_csv('Data/netflix_titles.csv')
print(df.shape)
print(df.head())
print(df.columns.tolist())
print(df.isnull().sum())

