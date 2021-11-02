from transform_data import *

# Amount of K-folds defined for outer, K1, and inner, K2 
K1 = 10
K2 = 10

# The outer cross-validation layer in the twofold cross validation is defined
cv_outer = KFold(n_splits=K1, shuffle=True, random_state=1)

for train_idxs, test_idxs in cv_outer.split(X):
    # split data into the K1 folds, train and test
    X_train, X_test = X[train_idxs, :], X[test_idxs, :]
    y_train, y_test = y[train_idxs], y[test_idxs]

    # The inner cross-validation layer in the twofold cross validation is defined
    cv_inner = KFold(n_splits=K2, shuffle=True, random_state=1)