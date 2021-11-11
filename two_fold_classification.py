# Some code is reused from the Toolbox given in the course (02450)

import numpy as np
import pandas as pd
import load_data
from sklearn import preprocessing
from sklearn.dummy import DummyClassifier
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

for feature in features:
    le = preprocessing.LabelEncoder()
    X[feature] = le.fit_transform(X[feature])

LABEL_ENCODED = pd.DataFrame(X.values, columns=X.columns)
X = LABEL_ENCODED.to_numpy()



# =============================================
# Baseline, Logistic Regression and Classification Tree two-fold cross-validation
# =============================================

# Amount of K-folds in inner and outer fold
K1 = 10
K2 = 10


# Logistic regression initialization
lambda_interval = np.logspace(-1, 3, 50)


# Classification tree initialization
# Tree complexity parameter range - constraint on tree maximum depth
tc = np.arange(2, 21, 1)


# Initialize error arrays
error_test_baseline = np.empty((K1,1))
error_test_lr = np.empty((K1,1))
error_test_ct = np.empty((K1,1))
opt_tree_comlexity = np.empty((K1,1))
opt_lambda_complexity = np.empty((K1,1))

# Outer cross validation layer
cv_outer = KFold(n_splits=K1, shuffle=True, random_state=1)
k1 = 0
for train_idxs, test_idxs in cv_outer.split(X,y):
    print("Outer cv: "  + str(k1+1) + "/" + str(K1))

    # Split data into the K1 folds, train and test
    X_train, X_test = X[train_idxs, :], X[test_idxs, :]
    y_train, y_test = y[train_idxs], y[test_idxs]

    # Controls the indices of the inner-layer variable loop-throughs (CT model)
    f = 0

    # Store test error data for CT
    test_error_ct = np.empty((K2, len(tc)))
    #Store test error data for LR
    test_error_lr = np.empty((K2, len(lambda_interval)))
    # Store mean square errors for the baseline model
    MSEs_baseline = []

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


        # ================ Inner CT model ================
        tc = np.arange(2, 21, 1)

        for i, t in enumerate(tc):
            # Fit decision tree classifier, Gini split criterion, different pruning levels
            dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
            dtc = dtc.fit(X_train_inner, y_train_inner.ravel())

            y_est_test = dtc.predict(X_test_inner)

            MSE = np.sum((y_test_inner - y_est_test) ** 2) / float(len(y_test_inner))
            test_error_ct[f,i] = MSE

        # =================================================

        # ================ Inner LR model ================
        train_error_rate = np.zeros(len(lambda_interval))
        test_error_rate = np.zeros(len(lambda_interval))
        coefficient_norm = np.zeros(len(lambda_interval))
        for k in range(0, len(lambda_interval)):
            model = LogisticRegression(penalty='l2', C=1 / lambda_interval[k],max_iter=1000)

            model.fit(X_train_inner, y_train_inner)

            y_train_est = model.predict(X_train_inner).T
            y_test_est = model.predict(X_test_inner).T

            train_error_rate[k] = np.sum(y_train_est != y_train_inner) / len(y_train_inner)
            test_error_rate[k] = np.sum(y_test_est != y_test_inner) / len(y_test_inner)

            w_est = model.coef_[0]
            coefficient_norm[k] = np.sqrt(np.sum(w_est ** 2))

        # Find optimal lambda value
        min_error = np.min(test_error_rate)
        opt_lambda_idx = np.argmin(test_error_rate)
        opt_lambda = lambda_interval[opt_lambda_idx]
        LR = LogisticRegression(penalty='l2', C=1/opt_lambda,max_iter=1000)
        LR.fit(X_train_inner,y_train_inner)
        y_est_test_lr = LR.predict(X_test_inner)

        MSE_LR = np.sum((y_test_inner - y_est_test_lr) ** 2) / float(len(y_test_inner))
        test_error_lr[f, opt_lambda_idx] = MSE_LR


        # =================================================

        f = f+1

        # ================ Inner Baseline model ================
        # Apply baseline model
        baseline_clf = DummyClassifier(strategy="most_frequent")
        baseline_clf.fit(X_train_inner, y_train_inner)
        predict_baseline = baseline_clf.predict(X_test_inner)

        # Calculate mean square loss
        MSE_baseline = np.sum((y_test_inner - predict_baseline)**2) / float(len(y_test_inner))
        MSEs_baseline.append(MSE_baseline)
        # =================================================



    # Baseline errors (outer layer)
    error_test_baseline[k1] = len(X_test_inner)/len(X_train) * np.sum(MSEs_baseline)


    # Extract optimal error and parameters (classification)
    opt_val_err_ct = np.min(np.mean(test_error_ct, axis=0))
    opt_tc = tc[np.argmin(np.mean(test_error_ct,axis=0))]


    # Extract optimal error and parameters (logistic regression)
    opt_val_err_lr = np.min(np.mean(test_error_lr, axis=0))
    opt_val_lambda = lambda_interval[np.argmin(np.mean(test_error_lr,axis=0))]


    # ========= Re-train CT with optimal parameters (outer layer) =========
    # Fit decision tree classifier, Gini split criterion, different pruning levels
    dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=opt_tc)
    dtc = dtc.fit(X_train, y_train.ravel())

    y_est_test = dtc.predict(X_test)

    # Calculate MSE with optimal parameters
    mse_ct = np.sum((y_test - y_est_test) ** 2) / float(len(y_test))

    # Save test error and optimal parameters for CT
    error_test_ct[k1] = mse_ct
    opt_tree_comlexity[k1] = opt_tc

    # ========= Re-train LR with optimal parameters (outer layer) =========
    LR = LogisticRegression(penalty='l2', C=1 / opt_val_lambda, max_iter=1000)
    LR.fit(X_train, y_train)
    y_est_test_lr = LR.predict(X_test)

    # Calculate MSE with optimal parameters
    mse_lr = np.sum((y_test - y_est_test_lr) ** 2) / float(len(y_test))

    # Save test error and optimal parameters for LR
    error_test_lr[k1] = mse_lr
    opt_lambda_complexity[k1] = opt_val_lambda


    k1 += 1





# Save data and prep data for export to latex

data = [opt_tree_comlexity.squeeze(),error_test_ct.squeeze(),opt_lambda_complexity.squeeze(),error_test_lr.squeeze(),error_test_baseline.squeeze()]
data = np.transpose(np.array(data))
print(data)

results_table = pd.DataFrame(data,columns=["Classification Tree"," ","Logistic Regression"," ","Baseline"])
results_table.index += 1

print(results_table.to_latex())

f = open("classification_latex_table.txt","w+")
f.write(results_table.to_latex())
f.close()