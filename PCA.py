import numpy as np
from numpy.lib.shape_base import expand_dims
import pandas as pd
from matplotlib import pyplot as plt
import load_data
import os
import seaborn as sns
from sklearn import preprocessing
from scipy.linalg import svd
# ignore warnings
import warnings
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

if not os.path.exists("plots"):
    os.mkdir("plots")
pd.options.mode.chained_assignment = None

# ------------------------- Label Encoding ------------------------------------------
df = load_data.df
X = df.drop(['annual-income'], axis=1)
y = df['annual-income']

features = ["education","sex","native-country","workclass","marital-status","occupation","relationship","race"]
for feature in features:
    le = preprocessing.LabelEncoder()
    X[feature] = le.fit_transform(X[feature])
# print(df)
# print(df.isna().sum().sum())

scaled_data = preprocessing.scale(X.values)
LABEL_ENCODED = pd.DataFrame(scaled_data, columns = X.columns)
# print(X)
# PCA and explained variance with label encoded data
N = LABEL_ENCODED.shape[0]
X = np.array(LABEL_ENCODED)
# Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(axis=0)
# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

# -------------------------------- One-Hot Encoding -------------------------
# ONE_HOT_ENCODED = pd.get_dummies(X, columns = features)
# # print(ONE_HOT_ENCODED.head())
#
# # PCA and explained variance with One-Hot Encoded data
# N = ONE_HOT_ENCODED.shape[0]
# X = np.array(ONE_HOT_ENCODED)
# # Subtract mean value from data
# Y = X - np.ones((N,1))*X.mean(axis=0)
# # PCA by computing SVD of Y
# U,S,V = svd(Y,full_matrices=False)
#
# # Compute variance explained by principal components
# rho = (S*S) / (S*S).sum()
#
# threshold = 0.9
#
# # Plot variance explained
# plt.figure()
# plt.plot(range(1,len(rho)+1),rho,'x-')
# plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
# plt.plot([1,len(rho)],[threshold, threshold],'k--')
# plt.title('Variance explained by principal components');
# plt.xlabel('Principal component');
# plt.ylabel('Variance explained');
# plt.legend(['Individual','Cumulative','Threshold'])
# plt.grid()
# plt.show()

