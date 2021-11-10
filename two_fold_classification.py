# Some code is reused from the Toolbox given in the course (02450)

import numpy as np
import pandas as pd
import load_data
from sklearn import preprocessing
from sklearn import tree,model_selection
from sklearn.model_selection import KFold
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show, boxplot
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from toolbox_02450 import rocplot, confmatplot


df = load_data.df
X = df.drop(['annual-income'], axis=1)
features = list(X.columns)
y = df['annual-income']
class_labels = df["annual-income"]
class_names = sorted(set(class_labels))
class_dict = dict(zip(class_names, range(2)))
y = np.asarray([class_dict[value] for value in class_labels])
X = X.to_numpy()



# =============================================
# Baseline, Logistic Regression and Classification Tree two-fold cross-validation
# =============================================

# Amount of K-folds in inner and outer fold
K1 = 10
K2 = 10

# Baseline initialization



# Logistic regression initialization


# ================  Classification tree initialization  ================


# Tree complexity parameter range - constraint on tree maximum depth
tc = np.arange(2, 21, 1)



# =============================================


# Initialize error arrays
error_test_baseline = np.empty((K1,1))
error_test_lr = np.empty((K1,1))
error_test_ct = np.empty((K1,1))
opt_tree_comlexity = np.empty((K1,1))

# Outer cross validation layer
cv_outer = KFold(n_splits=K1, shuffle=True, random_state=1)
k1 = 0
for train_idxs, test_idxs in cv_outer.split(X,y):
    print("Outer cv: "  + str(k1+1) + "/" + str(K1))

    # Split data into the K1 folds, train and test
    X_train, X_test = X[train_idxs, :], X[test_idxs, :]
    y_train, y_test = y[train_idxs], y[test_idxs]

    # Denotes the indices of the inner-layer variable loop-throughs
    f = 0

    test_error_ct = np.empty((K2, len(tc)))

    # y = y.squeeze()
    # X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)

    # INNER cross-validation layer
    cv_inner = KFold(n_splits=K2, shuffle=True, random_state=1)
    for train_idxs_inner, test_idx_inner in cv_outer.split(X_train,y_train):

        # Make the inner train and test data
        X_train_inner, X_test_inner = X_train[train_idxs_inner, :], X_train[test_idx_inner, :]
        y_train_inner, y_test_inner = y_train[train_idxs_inner], y_train[test_idx_inner]

        # STANDARDIZE inner training and test set
        mu_inner = np.mean(X_train_inner[:, 1:], 0)
        sigma_inner = np.std(X_train_inner[:, 1:], 0)

        X_train_inner[:, 1:] = (X_train_inner[:, 1:] - mu_inner) / sigma_inner
        X_test_inner[:, 1:] = (X_test_inner[:, 1:] - mu_inner) / sigma_inner


        # ================ Apply Classification model ================

        for feature in features:
            le = preprocessing.LabelEncoder()
            X_train_inner[feature] = le.fit_transform(X_train_inner[feature])

        LABEL_ENCODED = pd.DataFrame(X_train_inner.values, columns=X_train_inner.columns)
        X_train_inner = LABEL_ENCODED.to_numpy()

        # Class labels extraction
        class_labels = y_train_inner
        class_names = sorted(set(class_labels))
        class_dict = dict(zip(class_names, range(2)))

        # array denoting classes for all instances
        y_train_inner = np.asarray([class_dict[value] for value in class_labels])

        # Tree complexity parameter - constraint on maximum depth
        tc = np.arange(2, 21, 1)

        for i, t in enumerate(tc):
            # Fit decision tree classifier, Gini split criterion, different pruning levels
            dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
            dtc = dtc.fit(X_train_inner, y_train_inner.ravel())

            y_est_test = dtc.predict(X_test_inner)

            MSE = np.sum((y_test_inner - y_est_test) ** 2) / float(len(y_test_inner))
            test_error_ct[f,i] = MSE

        f = f+1

    # Extract optimal error and parameters (classification)
    opt_val_err_ct = np.min(np.mean(test_error_ct, axis=0))
    opt_tc = tc[np.argmin(np.mean(test_error_tc,axis=0))]

    #print(f'Optimal error (CT): {opt_val_err_ct}')
    #print(f"Optimal tree complexity (CT): {opt_tc}")


    # Re-train CT with optimal parameters (outer layer)
    #print(f"Retraining CT with optimal tc...")

    for feature in features:
        le = preprocessing.LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])

    LABEL_ENCODED = pd.DataFrame(X_train.values, columns=X_train.columns)
    X_train = LABEL_ENCODED.to_numpy()

    # Class labels extraction
    class_labels = y_train
    class_names = sorted(set(class_labels))
    class_dict = dict(zip(class_names, range(2)))

    # Should the Class label extraction and Label Encoding happen before splitting data?

    # array denoting classes for all instances
    y_train = np.asarray([class_dict[value] for value in class_labels])

    # Fit decision tree classifier, Gini split criterion, different pruning levels
    dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=opt_tc)
    dtc = dtc.fit(X_train, y_train.ravel())

    y_est_test = dtc.predict(X_test_inner)

    # Calculate MSE with optimal parameters
    mse_ct = np.sum((y_test_inner - y_est_test) ** 2) / float(len(y_test_inner))

    # Save test error and optimal parameters for CT
    error_test_ct[k1] = mse_ct
    opt_tree_comlexity[k1] = opt_tc


    k1 += 1







data = [opt_tree_comlexity.squeeze(),error_test_ct.squeeze()]
data = np.transpose(np.array(data))
print(data)