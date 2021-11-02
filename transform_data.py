import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import load_data
import os
from sklearn import preprocessing
from sklearn.model_selection import KFold
import matplotlib as mpl
mpl.style.use("seaborn")

if not os.path.exists("plots"):
    os.mkdir("plots")
pd.options.mode.chained_assignment = None

pd.options.display.float_format = '{:.2f}'.format

# Get the dataframe from load_data
df = load_data.df

# Define features and label
X = df.drop(['annual-income'], axis=1)
y = df['annual-income']

# Get attribute names 
attribute_names = list(X.columns)

# Class labels extraction
class_labels = df["annual-income"]
class_names = sorted(set(class_labels))
class_dict = dict(zip(class_names, range(2)))

# y is created 
y = np.asarray([class_dict[value] for value in class_labels])

# The values of N, M and C are encoded
N = len(y)
M = len(attribute_names)
C = len(class_names)

# The label encoder is used on the data
for attribute_name in attribute_names:
    le = preprocessing.LabelEncoder()
    X[attribute_name] = le.fit_transform(X[attribute_name])

# Standadizing the data
scaled_data = preprocessing.scale(X.values)
# Create new dataframe with the data 
scaled_X_df = pd.DataFrame(scaled_data, columns = X.columns)

X = scaled_data