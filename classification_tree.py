# Some code is reused from the Toolbox given in the course (02450)

import numpy as np
import pandas as pd
import load_data
from sklearn import preprocessing
from sklearn import tree,model_selection
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show, boxplot
from sklearn.metrics import accuracy_score


df = load_data.df
X = df.drop(['annual-income'], axis=1)
y = df['annual-income']

features = list(X.columns)
# Label encoding for all the data features. Scaling is not required since Classification trees aren't sensitive to scaling
for feature in features:
    le = preprocessing.LabelEncoder()
    X[feature] = le.fit_transform(X[feature])

LABEL_ENCODED = pd.DataFrame(X.values, columns = X.columns)
X = LABEL_ENCODED.to_numpy()
# print(X.shape)

# Class labels extraction
class_labels = df["annual-income"]
class_names = sorted(set(class_labels))
class_dict = dict(zip(class_names, range(2)))

# array denoting classes for all instances
y = np.asarray([class_dict[value] for value in class_labels])

# Tree complexity parameter - constraint on maximum depth
tc = np.arange(2, 21, 1)

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K, shuffle=True)

# Initialize variable
Error_train = np.empty((len(tc), K))
Error_test = np.empty((len(tc), K))

k = 0
for train_index, test_index in CV.split(X):
    print('Computing CV fold: {0}/{1}..'.format(k + 1, K))

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index, :], y[train_index]
    X_test, y_test = X[test_index, :], y[test_index]

    for i, t in enumerate(tc):
        # Fit decision tree classifier, Gini split criterion, different pruning levels
        dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
        dtc = dtc.fit(X_train, y_train.ravel())
        y_est_test = dtc.predict(X_test)
        y_est_train = dtc.predict(X_train)
        # Evaluate misclassification rate over train/test data (in this CV fold)
        misclass_rate_test = np.sum(y_est_test != y_test) / float(len(y_est_test))
        misclass_rate_train = np.sum(y_est_train != y_train) / float(len(y_est_train))
        Error_test[i, k], Error_train[i, k] = misclass_rate_test, misclass_rate_train
    k += 1

f = figure()
boxplot(Error_test.T)
xlabel('Model complexity (max tree depth)')
ylabel('Test error across CV folds, K={0})'.format(K))

f = figure()
plot(tc, Error_train.mean(1))
plot(tc, Error_test.mean(1))
xlabel('Model complexity (max tree depth)')
ylabel('Error (misclassification rate, CV K={0})'.format(K))
legend(['Error_train', 'Error_test'])

show()


test_proportion = 0.25
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_proportion)

# Optimal tree complexity using simple holdout-set crossvalidation and K-fold crossvalidation: tc = 7
dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=7)
dtc = dtc.fit(X_train, y_train)
y_est_test = np.asarray(dtc.predict(X_test), dtype=int)
# Accuracy score on the the test data (25%)
print(f"accuracy score: {accuracy_score(y_test,y_est_test)}")


# Simple holdout-set crossvalidation
# # Initialize variables
# Error_train = np.empty((len(tc), 1))
# Error_test = np.empty((len(tc), 1))
#
# for i, t in enumerate(tc):
#     # Fit decision tree classifier, Gini split criterion, different pruning levels
#     dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
#     dtc = dtc.fit(X_train, y_train)
#
#     # Evaluate classifier's misclassification rate over train/test data
#     y_est_test = np.asarray(dtc.predict(X_test), dtype=int)
#     y_est_train = np.asarray(dtc.predict(X_train), dtype=int)
#     misclass_rate_test = sum(y_est_test != y_test) / float(len(y_est_test))
#     misclass_rate_train = sum(y_est_train != y_train) / float(len(y_est_train))
#     Error_test[i], Error_train[i] = misclass_rate_test, misclass_rate_train
#
# f = figure()
# plot(tc, Error_train * 100)
# plot(tc, Error_test * 100)
# xlabel('Model complexity (max tree depth)')
# ylabel('Error (%)')
# legend(['Error_train', 'Error_test'])
#
# show()

