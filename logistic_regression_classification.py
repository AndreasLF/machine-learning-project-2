# exercise 8.1.2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from toolbox_02450 import rocplot, confmatplot
import load_data

df = load_data.df
X = df.drop(['annual-income'], axis=1)
#attribute_names = ["age", "capital-gain", "capital-loss", "hours-per-week", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
#X = df[attribute_names].to_numpy()

#y = df['annual-income']
class_labels = df["annual-income"]
class_names = sorted(set(class_labels))
class_dict = dict(zip(class_names, range(2)))
y = np.asarray([class_dict[value] for value in class_labels])

N, M = X.shape
C = len(class_names)

features = list(X.columns)
for feature in features:
    le = preprocessing.LabelEncoder()
    X[feature] = le.fit_transform(X[feature])

# Standardizing the data with
scaled_data = preprocessing.scale(X.values)
LABEL_ENCODED = pd.DataFrame(scaled_data, columns = X.columns)

X = LABEL_ENCODED.to_numpy()
print(X.shape,y.shape)
print(N,M,C)

# Create crossvalidation partition for evaluation
# using stratification and 90/10 pct. split between respectively training and test 
K = 20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, stratify=y)

# Standardize the training and set set based on training set mean and std
mu = np.mean(X_train, 0)
sigma = np.std(X_train, 0)

X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

# Fit regularized logistic regression model to training data to predict the type of 
lambda_interval = np.logspace(-6, 6, 50)
train_error_rate = np.zeros(len(lambda_interval))
test_error_rate = np.zeros(len(lambda_interval))
coefficient_norm = np.zeros(len(lambda_interval))
for k in range(0, len(lambda_interval)):
    model = LogisticRegression(penalty='l2', C=1/lambda_interval[k] )
    
    model.fit(X_train, y_train)

    y_train_est = model.predict(X_train).T
    y_test_est = model.predict(X_test).T
    
    train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
    test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)

    w_est = model.coef_[0] 
    coefficient_norm[k] = np.sqrt(np.sum(w_est**2))


#Find optimal lambda value
min_error = np.min(test_error_rate)
opt_lambda_idx = np.argmin(test_error_rate)
opt_lambda = lambda_interval[opt_lambda_idx]

#model1 = LogisticRegression(penalty='l2', C=1/2023)
model1 = LogisticRegression(penalty='l2', C=1/opt_lambda)
model1.fit(X_train, y_train)
y_pred_log = model1.predict(X_test)
print("accuracy logistic: ", accuracy_score(y_pred_log, y_test))
print("min error: ", min_error)

#Denne bruger den nyeste lambdaværdi ved 10^6 som har en dårlig accuracy
#y_test_est = model.predict(X_test)
#print("accuracy logistic OG: ", accuracy_score(y_test_est, y_test))

plt.figure(figsize=(8,8))
#plt.plot(np.log10(lambda_interval), train_error_rate*100)
#plt.plot(np.log10(lambda_interval), test_error_rate*100)
#plt.plot(np.log10(opt_lambda), min_error*100, 'o')
plt.semilogx(lambda_interval, train_error_rate*100)
plt.semilogx(lambda_interval, test_error_rate*100)
plt.semilogx(opt_lambda, min_error*100, 'o')
plt.text(1e-8, 3, "Minimum test error: " + str(np.round(min_error*100,2)) + ' % at 1e' + str(np.round(np.log10(opt_lambda),2)))
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.ylabel('Error rate (%)')
plt.title('Classification error')
plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
plt.ylim([0, 30])
plt.grid()
plt.savefig("plots2/log_reg_classi_"+ "test_error" +".jpg")


plt.figure(figsize=(8,8))
plt.semilogx(lambda_interval, coefficient_norm,'k')
plt.ylabel('L2 Norm')
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.title('Parameter vector L2 norm')
plt.grid()
plt.savefig("plots2/log_reg_classi_"+ "regularization" +".jpg")


#########################
#BASELINE
baseline_clf = DummyClassifier(strategy="most_frequent")
baseline_clf.fit(X_train, y_train)

y_train_est_base = baseline_clf.predict(X_train).T
y_test_est_base = baseline_clf.predict(X_test).T

print("\n base train predictions: ", y_train_est_base)
print("baseline training accuracy: ", baseline_clf.score(X,y))
print("baseline accuracy accuracy: ", accuracy_score(y_test, y_test_est_base))


 