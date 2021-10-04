import numpy as np
from numpy.lib.shape_base import expand_dims
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
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
import matplotlib as mpl
mpl.style.use("seaborn")

if not os.path.exists("plots"):
    os.mkdir("plots")
pd.options.mode.chained_assignment = None

pd.options.display.float_format = '{:.2f}'.format


# ------------------------- Label Encoding ------------------------------------------
df = load_data.df
X = df.drop(['annual-income'], axis=1)
y = df['annual-income']

# features = ["education","sex","native-country","workclass","marital-status","occupation","relationship","race"]
features = list(X.columns)

# Replace NaN values with a string 
# df = df.fillna('NaN')

# print(df.isna().sum())
# print()
for feature in features:
    le = preprocessing.LabelEncoder()
    X[feature] = le.fit_transform(X[feature])

# Standadizing the data with
scaled_data = preprocessing.scale(X.values)
LABEL_ENCODED = pd.DataFrame(scaled_data, columns = X.columns)

# -----------------------------------------------------------------------------------

# PCA and explained variance with label encoded data
N, M = LABEL_ENCODED.shape
X = np.array(LABEL_ENCODED)
# Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(axis=0)
# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)


# We saw in 2.1.3 that the first 3 components explaiend more than 90
# percent of the variance. Let's look at their coefficients:
pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, features)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Adult Census: PCA Component Coefficients')
plt.xticks(rotation=70)
plt.grid()
plt.savefig("plots/pca_coefficient_orientaion.jpg", bbox_inches='tight')


# ----------------------------------
# Plot explained variance
# ----------------------------------
# Compute explained variance
rho = (S*S) / (S*S).sum()
cummulative = np.cumsum(rho)
threshold = 0.9
# Clear plot 
plt.clf()
# Make plot 
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Explained Variance');
plt.xlabel('Principal components');
plt.ylabel('Percentage of explained variance');
plt.legend(['Individual','Cumulative','Threshold'])
plt.savefig("plots/explained_variance_plot.jpg", bbox_inches='tight')


# Project the centered data onto principal component space
Z = Y @ V

classes = sorted(set(y))
class_dict = dict(zip(classes,range(2)))
y = np.asarray([class_dict[value] for value in y])

C = len(classes)

# ----------------------------------
# Plot principal components
# ----------------------------------
# Indices of the principal components to be plotted
PCs_list = [(0,1),(1,2),(11,12)]

for PCs in PCs_list:
    i = PCs[0]
    j = PCs[1]

    # Clear plot 
    plt.clf()
    # Plot PCA of the data
    f = figure()
    title('Adult Census: PCA')
    for c in range(C):
        # select indices belonging to class c:
        class_mask = y==c
        plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
    legend(classes)
    xlabel('PC{0}'.format(i+1))
    ylabel('PC{0}'.format(j+1))
    f.savefig("plots/PC"+str(i+1)+"_PC"+str(j+1)+"_plot.jpg", bbox_inches='tight')

principal_components = []
for n in range(1,15):
    principal_components.append(f'PC{n}')
# print(principal_components)
ExplainedVarTable = pd.DataFrame(np.matrix([rho*100,cummulative*100]).T,
                                 index=principal_components,
                                 columns=["Explained Variance","Cummulative Sum"])
print(ExplainedVarTable.to_latex())












# ----------------------- PCA Directions --------------------------
# # Subtract the mean from the data
# Y1 = X - np.ones((N, 1)) * X.mean(0)
#
# # Subtract the mean from the data and divide by the attribute standard
# # deviation to obtain a standardized dataset:
# Y2 = X - np.ones((N, 1)) * X.mean(0)
# Y2 = Y2 * (1 / np.std(Y2, 0))
# # Here were utilizing the broadcasting of a row vector to fit the dimensions
# # of Y2
#
# # Store the two in a cell, so we can just loop over them:
# Ys = [Y1, Y2]
# titles = ['Zero-mean', 'Zero-mean and unit variance']
# threshold = 0.9
# # Choose two PCs to plot (the projection)
# i = 0
# j = 1
#
# # Make the plot
# plt.figure(figsize=(10, 15))
# plt.subplots_adjust(hspace=.4)
# plt.title('NanoNose: Effect of standardization')
# nrows = 3
# ncols = 2
# for k in range(2):
#     # Obtain the PCA solution by calculate the SVD of either Y1 or Y2
#     U, S, Vh = svd(Ys[k], full_matrices=False)
#     V = Vh.T  # For the direction of V to fit the convention in the course we transpose
#     # For visualization purposes, we flip the directionality of the
#     # principal directions such that the directions match for Y1 and Y2.
#     if k == 1: V = -V; U = -U;
#
#     # Compute variance explained
#     rho = (S * S) / (S * S).sum()
#
#     # Compute the projection onto the principal components
#     Z = U * S;
#
#     # Plot projection
#     plt.subplot(nrows, ncols, 1 + k)
#     C = len(classes)
#     for c in range(C):
#         plt.plot(Z[y == c, i], Z[y == c, j], '.', alpha=.5)
#     plt.xlabel('PC' + str(i + 1))
#     plt.xlabel('PC' + str(j + 1))
#     plt.title(titles[k] + '\n' + 'Projection')
#     plt.legend(classes)
#     plt.axis('equal')
#
#     # Plot attribute coefficients in principal component space
#     plt.subplot(nrows, ncols, 3 + k)
#     for att in range(V.shape[1]):
#         plt.arrow(0, 0, V[att, i], V[att, j])
#         plt.text(V[att, i], V[att, j], features[att]) 
#     plt.xlim([-1, 1])
#     plt.ylim([-1, 1])
#     plt.xlabel('PC' + str(i + 1))
#     plt.ylabel('PC' + str(j + 1))
#     plt.grid()
#     # Add a unit circle
#     plt.plot(np.cos(np.arange(0, 2 * np.pi, 0.01)),
#              np.sin(np.arange(0, 2 * np.pi, 0.01)));
#     plt.title(titles[k] + '\n' + 'Attribute coefficients')
#     plt.axis('equal')
#
#     # Plot cumulative variance explained
#     plt.subplot(nrows, ncols, 5 + k);
#     plt.plot(range(1, len(rho) + 1), rho, 'x-')
#     plt.plot(range(1, len(rho) + 1), np.cumsum(rho), 'o-')
#     plt.plot([1, len(rho)], [threshold, threshold], 'k--')
#     plt.title('Variance explained by principal components');
#     plt.xlabel('Principal component');
#     plt.ylabel('Variance explained');
#     plt.legend(['Individual', 'Cumulative', 'Threshold'])
#     plt.grid()
#     plt.title(titles[k] + '\n' + 'Variance explained')
#
# plt.show()

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

