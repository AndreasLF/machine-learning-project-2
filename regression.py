from transform_data import *
import sklearn.linear_model as lm
from sklearn import model_selection
import torch
from toolbox_02450 import train_neural_net

# Check if graphics card is available 
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
device = torch.device(dev)  

# =============================================
# Prepare data for regression
# =============================================
# Load data 
df = load_data.df

# Define features and label
X = df.drop(['hours-per-week'], axis=1)
y = df['hours-per-week']

# Get attribute names 
attribute_names = list(X.columns)

# The values of N, M are encoded
N = len(y)
M = len(attribute_names)

# The label encoder is used on the data
for attribute_name in attribute_names:
    le = preprocessing.LabelEncoder()
    X[attribute_name] = le.fit_transform(X[attribute_name])

scaled_data = preprocessing.scale(X.values)
# Create new dataframe with the data 
scaled_X_df = pd.DataFrame(scaled_data, columns = X.columns)


# X = scaled_data
X = X.to_numpy()
y = y.to_numpy()

print(X)
# =============================================
# Check if matrix is invertible 
# =============================================

xx= np.matmul(np.transpose(X),X)

if (np.linalg.det(xx)):
    print("Non Singular matrix")
else:
    print("Singular Matrix")

I = np.around(np.matmul(xx,np.linalg.inv(xx)),decimals=0)
print(I)
# It has an inverse. No need to use regularization?


# =============================================
# Linear regression model with residuals plot
# =============================================

# Fit ordinary least squares regression model
model = lm.LinearRegression(fit_intercept=True)
model = model.fit(X,y)
# Compute model output:
y_est = model.predict(X)
# Or equivalently:
#y_est = model.intercept_ + X @ model.coef_

residual = y_est-y

# Display scatter plot
plt.subplot(2,1,1)
plt.plot(y, y_est, '.')
plt.xlabel('hours-per-week (true)'); 
plt.ylabel('hours-per-week (estimated)');
plt.subplot(2,1,2)
plt.hist(residual,40)


plt.tight_layout()

filename = 'plots2/linar_regression_residuals.jpg'
if os.path.isfile(filename):
   os.remove(filename)
plt.savefig(filename)

# =============================================
# Regularization
# =============================================
def rlr_validate(X,y,lambdas,cvf=10):
    ''' Validate regularized linear regression model using 'cvf'-fold cross validation.
        Find the optimal lambda (minimizing validation error) from 'lambdas' list.
        The loss function computed as mean squared error on validation set (MSE).
        Function returns: MSE averaged over 'cvf' folds, optimal value of lambda,
        average weight values for all lambdas, MSE train&validation errors for all lambdas.
        The cross validation splits are standardized based on the mean and standard
        deviation of the training set when estimating the regularization strength.
        
        Parameters:
        X       training data set
        y       vector of values
        lambdas vector of lambda values to be validated
        cvf     number of crossvalidation folds     
        
        Returns:
        opt_val_err         validation error for optimum lambda
        opt_lambda          value of optimal lambda
        mean_w_vs_lambda    weights as function of lambda (matrix)
        train_err_vs_lambda train error as function of lambda (vector)
        test_err_vs_lambda  test error as function of lambda (vector)
    '''
    CV = model_selection.KFold(cvf, shuffle=True, random_state=1)
    M = X.shape[1]
    w = np.empty((M,cvf,len(lambdas)))
    train_error = np.empty((cvf,len(lambdas)))
    test_error = np.empty((cvf,len(lambdas)))
    f = 0
    y = y.squeeze()
    for train_index, test_index in CV.split(X,y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        
        # Standardize the training and set set based on training set moments
        mu = np.mean(X_train[:, 1:], 0)
        sigma = np.std(X_train[:, 1:], 0)
        
        X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
        X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma

        # precompute terms
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train
        for l in range(0,len(lambdas)):
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            lambdaI = lambdas[l] * np.eye(M)
            lambdaI[0,0] = 0 # remove bias regularization
            w[:,f,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
            # Evaluate training and test performance
            train_error[f,l] = np.power(y_train-X_train @ w[:,f,l].T,2).mean(axis=0)
            test_error[f,l] = np.power(y_test-X_test @ w[:,f,l].T,2).mean(axis=0)
    
        f=f+1

    opt_val_err = np.min(np.mean(test_error,axis=0))
    opt_lambda = lambdas[np.argmin(np.mean(test_error,axis=0))]
    train_err_vs_lambda = np.mean(train_error,axis=0)
    test_err_vs_lambda = np.mean(test_error,axis=0)
    mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))
    
    return opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda
       
# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attribute_names
M = M+1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 5
CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambdas = np.power(10.,range(-2,12))

# Initialize variables
#T = len(lambdas)
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
error_train_rlr = np.empty((K,1))
error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))

k=0
for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10    
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    #m = lm.LinearRegression().fit(X_train, y_train)
    #Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    #Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Display the results for the last cross-validation fold
    if k == K-1:
        plt.figure(k, figsize=(12,8))
        plt.subplot(1,2,1)
        plt.semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        plt.xlabel('Regularization factor')
        plt.ylabel('Mean Coefficient Values')
        plt.grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        plt.legend(attributeNames[1:], loc='best')
        
        plt.subplot(1,2,2)
        plt.title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        plt.loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        plt.xlabel('Regularization factor')
        plt.ylabel('Squared error (crossvalidation)')
        plt.legend(['Train error','Validation error'])
        plt.grid()
    
    # # To inspect the used indices, use these print statements
    # print('Cross validation fold {0}/{1}:'.format(k+1,K))
    # print('Train indices: {0}'.format(train_index))
    # print('Test indices: {0}\n'.format(test_index))

    k+=1

plt.tight_layout()
filename = 'plots2/regularization_error.jpg'

if os.path.isfile(filename):
   os.remove(filename)
plt.savefig(filename)
# plt.show()

# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(error_train_rlr.mean()))
print('- Test error:     {0}'.format(error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-error_test_rlr.sum())/Error_test_nofeatures.sum()))


print("Optimal lambda: ", opt_lambda)
print()

print('Weights in last fold:')
table = pd.DataFrame(columns=["Attributes", "Model weights in las fold (with regularization)", "Weights (without regularization)"])
table["Attributes"] = attributeNames
table.index = table["Attributes"]
table = table.drop("Attributes",axis=1)

for m in range(M):
    table["Model weights in las fold (with regularization)"][attributeNames[m]] = np.round(w_rlr[m,-1],2)
    table["Weights (without regularization)"][attributeNames[m]] = np.round(w_noreg[m,-1],2)

    # print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))

# Print latex table 
print(table.to_latex())


# =============================================
# Regression and baseline two-fold cross-validation
# =============================================

# Amount of K-folds in inner and outer fold
K1 = 10
K2 = 10


# Values of lambda
lambdas = np.power(10.,range(-2,12))

# Initialize variables 


error_train_rlr = np.empty((K1,1))
error_test_rlr = np.empty((K1,1))
opt_lambdas = np.empty((K1,1))
error_test_baseline = np.empty((K1,1))

w_rlr = np.empty((M,K1))

mu = np.empty((K1, M-1))
sigma = np.empty((K1, M-1))

# Parameters for neural network classifier
max_iter = 200
tolerance=1e-6
loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
# Initialize variables 
hidden_units = range(1,5)
error_test_ann = np.empty((K1,1))
opt_hidden_units_ann = np.empty((K1,1))

# ===========================================================================
# OUTER cross-validation layer
cv_outer = KFold(n_splits=K1, shuffle=True, random_state=1)
k1 = 0
for train_idxs, test_idxs in cv_outer.split(X,y):
    print("Outer cv: "  + str(k1+1) + "/" + str(K1))

    # split data into the K1 folds, train and test
    X_train, X_test = X[train_idxs, :], X[test_idxs, :]
    y_train, y_test = y[train_idxs], y[test_idxs]


    # Define lists to keep 
    MSEs_baseline = []
    MSEs_regression = []
    MSEs_ann = []

    M = X.shape[1]
    w = np.empty((M,K2,len(lambdas)))
    train_error = np.empty((K2,len(lambdas)))
    test_error = np.empty((K2,len(lambdas)))
    f = 0
    y = y.squeeze()


    # Store test error ANN
    test_error_ann = np.empty((K2,len(hidden_units)))

    # --------------------------------------------------
    # INNER cross-validation layer
    cv_inner = KFold(n_splits=K2, shuffle=True, random_state=1)
    k2 = 0
    for train_idxs_inner, test_idx_inner in cv_outer.split(X_train,y_train): 
        # print("Inner cv: "  + str(k2+1) + "/" + str(K2))

        # Make the inner train and test data 
        X_train_inner, X_test_inner = X_train[train_idxs_inner, :], X_train[test_idx_inner, :]
        y_train_inner, y_test_inner = y_train[train_idxs_inner], y_train[test_idx_inner]
        
        # STANDARDIZE inner training and test set
        mu_inner = np.mean(X_train_inner[:, 1:], 0)
        sigma_inner = np.std(X_train_inner[:, 1:], 0)

        X_train_inner[:, 1:] = (X_train_inner[:, 1:] - mu_inner) / sigma_inner
        X_test_inner[:, 1:] = (X_test_inner[:, 1:] - mu_inner) / sigma_inner

        # precompute terms
        Xty = X_train_inner.T @ y_train_inner
        XtX = X_train_inner.T @ X_train_inner


        # reshape labels to correspond to the dimensions of the train data
        y_train_inner_r = y_train_inner.reshape(len(y_train_inner),1)  
        y_test_inner_r = y_test_inner.reshape(len(y_test_inner),1)   

        # Convert data to tensors 
        X_train_inner_tensor = torch.from_numpy(X_train_inner).to(device)
        y_train_inner_tensor = torch.from_numpy(y_train_inner_r).to(device)
        X_test_inner_tensor = torch.from_numpy(X_test_inner).to(device)
        y_test_inner_tensor = torch.from_numpy(y_test_inner_r).to(device)

        # Convert tensors to float tensorst
        X_train_inner_tensor =  X_train_inner_tensor.type(torch.FloatTensor)
        y_train_inner_tensor =  y_train_inner_tensor.type(torch.FloatTensor)
        X_test_inner_tensor =  X_test_inner_tensor.type(torch.FloatTensor)
        y_test_inner_tensor =  y_test_inner_tensor.type(torch.FloatTensor)

        # Loop through lambdas 
        for l in range(0,len(lambdas)):
            # Compute parameters for current value of lambda and current CV fold
            lambdaI = lambdas[l] * np.eye(M)
            lambdaI[0,0] = 0 # remove bias regularization
            w[:,f,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
            # Evaluate training and test performance
            train_error[f,l] = np.power(y_train_inner-X_train_inner @ w[:,f,l].T,2).mean(axis=0)
            test_error[f,l] = np.power(y_test_inner-X_test_inner @ w[:,f,l].T,2).mean(axis=0)
        
        for key, n_hidden_units in enumerate(hidden_units): 
            print('\n\Hidden units: {}'.format(n_hidden_units))

            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                                # no final tranfer function, i.e. "linear output"
                                )

            best_net, best_final_loss, best_learning_curve = train_neural_net(model, loss_fn, X_train_inner_tensor, y_train_inner_tensor,n_replicates=1, max_iter=max_iter,tolerance=tolerance)
            print('\n\tBest loss: {}\n'.format(best_final_loss))
            
            # Make predictions 
            y_test_est = best_net(X_test_inner_tensor)
    
            # Determine errors and errors
            se = (y_test_est.float()-y_test_inner_tensor.float())**2 # squared error
            mse = (sum(se).type(torch.float)/len(y_test_inner_tensor)).data.numpy() #mean
            
            test_error_ann[f,key] = mse
        f=f+1

        # print(train_idxs_inner.shape)

        # Apply baseline model. Mean of train labels
        predict_baseline = np.mean(y_train_inner)
      
        # Calculate mean square loss 
        MSE_baseline = np.sum((y_test_inner - predict_baseline)**2) / float(len(y_test_inner)) 
        MSEs_baseline.append(MSE_baseline)


        # k2 += 1
    # --------------------------------------------------
    
    opt_val_err = np.min(np.mean(test_error,axis=0))
    opt_lambda = lambdas[np.argmin(np.mean(test_error,axis=0))]
    train_err_vs_lambda = np.mean(train_error,axis=0)
    test_err_vs_lambda = np.mean(test_error,axis=0)
    mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k1, :] = np.mean(X_train[:, 1:], 0)
    sigma[k1, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k1, :] ) / sigma[k1, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k1, :] ) / sigma[k1, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k1] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    error_train_rlr[k1] = np.square(y_train-X_train @ w_rlr[:,k1]).sum(axis=0)/y_train.shape[0]
    error_test_rlr[k1] = np.square(y_test-X_test @ w_rlr[:,k1]).sum(axis=0)/y_test.shape[0]
    opt_lambdas[k1] = opt_lambda

    error_test_baseline[k1] = len(X_test_inner)/len(X_train) * np.sum(MSEs_baseline)

    opt_val_err_ann = np.min(np.mean(test_error_ann,axis=0))
    opt_hidden_units = hidden_units[np.argmin(np.mean(test_error_ann,axis=0))]

    print("Optimal error", opt_val_err_ann)
    print("Optimal amount of hidden units", opt_hidden_units)

    # reshape labels to correspond to the dimensions of the train data
    y_train_r = y_train.reshape(len(y_train),1)  
    y_test_r = y_test.reshape(len(y_test),1)   

    # Convert data to tensors 
    X_train_tensor = torch.from_numpy(X_train).to(device)
    y_train_tensor = torch.from_numpy(y_train_r).to(device)
    X_test_tensor = torch.from_numpy(X_test).to(device)
    y_test_tensor = torch.from_numpy(y_test_r).to(device)

    # Convert tensors to float tensorst
    X_train_tensor =  X_train_tensor.type(torch.FloatTensor)
    y_train_tensor =  y_train_tensor.type(torch.FloatTensor)
    X_test_tensor =  X_test_tensor.type(torch.FloatTensor)
    y_test_tensor =  y_test_tensor.type(torch.FloatTensor)

    # Create new model with optimal hidden units 
    model = lambda: torch.nn.Sequential(
                                    torch.nn.Linear(M, opt_hidden_units), #M features to n_hidden_units
                                    torch.nn.Tanh(),   # 1st transfer function,
                                    torch.nn.Linear(opt_hidden_units, 1), # n_hidden_units to 1 output neuron
                                    # no final tranfer function, i.e. "linear output"
                                    )

    # Run ANN with this model
    best_net, best_final_loss, best_learning_curve = train_neural_net(model, loss_fn, X_train_tensor, y_train_tensor,n_replicates=1, max_iter=max_iter,tolerance=tolerance)

    y_test_est = best_net(X_test_tensor)
    
    # Determine errors and errors
    se = (y_test_est.float()-y_test_tensor.float())**2 # squared error
    mse = (sum(se).type(torch.float)/len(y_test_tensor)).data.numpy() #mean
    # errors.append(mse) # store error rate for current CV fold 
    print(mse)
    # Save test error and optimal hidden layers for ANN
    opt_hidden_units_ann[k1] = opt_hidden_units
    error_test_ann[k1] = mse
    # Increment k-fold layer counter 
    break
    k1 += 1
# ===========================================================================



# error_test_ann = np.arange(K1)
# hidden_layers_ann = np.zeros(K1)

# opt_hidden_units_ann

data = [opt_hidden_units_ann.squeeze(), error_test_ann.squeeze() ,opt_lambdas.squeeze() ,error_test_rlr.squeeze() ,error_test_baseline.squeeze()]
data = np.transpose(np.array(data))

# print(data)
results_table = pd.DataFrame(data,columns=["ANN", " ", "Linear Regresion", "  ", "Baseline"])
results_table.index += 1 

print(results_table.to_latex()) 

f = open("regression_latex_table.txt","w+")
f.write(results_table.to_latex())
f.close()

# # =============================================
# # ANN
# # =============================================
# # Amount of K-folds in inner and outer fold
# K1 = 10
# K2 = 10

# error_train_rlr = np.empty((K1,1))
# error_test_rlr = np.empty((K1,1))
# opt_lambdas = np.empty((K1,1))
# error_test_baseline = np.empty((K1,1))

# # Parameters for neural network classifier
# max_iter = 200
# tolerance=1e-6
# loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
# # Initialize variables 
# hidden_units = range(1,5)
# error_test_ann = np.empty((K1,1))
# opt_hidden_units_ann = np.empty((K1,1))

# w_rlr = np.empty((M,K1))

# mu = np.empty((K1, M-1))
# sigma = np.empty((K1, M-1))

# # ===========================================================================
# # OUTER cross-validation layer
# cv_outer = KFold(n_splits=K1, shuffle=True, random_state=1)
# k1 = 0
# for train_idxs, test_idxs in cv_outer.split(X,y):
#     print("Outer cv: "  + str(k1+1) + "/" + str(K1))

#     # split data into the K1 folds, train and test
#     X_train, X_test = X[train_idxs, :], X[test_idxs, :]
#     y_train, y_test = y[train_idxs], y[test_idxs]

#     # Define lists to keep 
#     MSEs_ann = []

#     M = X.shape[1]
#     f = 0
#     y = y.squeeze()



#     # --------------------------------------------------
#     # INNER cross-validation layer
#     cv_inner = KFold(n_splits=K2, shuffle=True, random_state=1)
#     k2 = 0

#     # Store test error 
#     test_error_ann = np.empty((K2,len(hidden_units)))

#     f = 0
#     for train_idxs_inner, test_idx_inner in cv_outer.split(X_train,y_train): 
#         # print("Inner cv: "  + str(k2+1) + "/" + str(K2))

#         # Make the inner train and test data 
#         X_train_inner, X_test_inner = X_train[train_idxs_inner, :], X_train[test_idx_inner, :]
#         y_train_inner, y_test_inner = y_train[train_idxs_inner], y_train[test_idx_inner]
        
#         # STANDARDIZE inner training and test set
#         mu_inner = np.mean(X_train_inner[:, 1:], 0)
#         sigma_inner = np.std(X_train_inner[:, 1:], 0)

#         X_train_inner[:, 1:] = (X_train_inner[:, 1:] - mu_inner) / sigma_inner
#         X_test_inner[:, 1:] = (X_test_inner[:, 1:] - mu_inner) / sigma_inner

#         # reshape labels to correspond to the dimensions of the train data
#         y_train_inner_r = y_train_inner.reshape(len(y_train_inner),1)  
#         y_test_inner_r = y_test_inner.reshape(len(y_test_inner),1)   

#         # Convert data to tensors 
#         X_train_inner_tensor = torch.from_numpy(X_train_inner)
#         y_train_inner_tensor = torch.from_numpy(y_train_inner_r)
#         X_test_inner_tensor = torch.from_numpy(X_test_inner)
#         y_test_inner_tensor = torch.from_numpy(y_test_inner_r)

#         # Convert tensors to float tensorst
#         X_train_inner_tensor =  X_train_inner_tensor.type(torch.FloatTensor)
#         y_train_inner_tensor =  y_train_inner_tensor.type(torch.FloatTensor)
#         X_test_inner_tensor =  X_test_inner_tensor.type(torch.FloatTensor)
#         y_test_inner_tensor =  y_test_inner_tensor.type(torch.FloatTensor)


#         for key, n_hidden_units in enumerate(hidden_units): 
#             print('\n\Hidden units: {}'.format(n_hidden_units))

#             model = lambda: torch.nn.Sequential(
#                                 torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
#                                 torch.nn.Tanh(),   # 1st transfer function,
#                                 torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
#                                 # no final tranfer function, i.e. "linear output"
#                                 )

#             best_net, best_final_loss, best_learning_curve = train_neural_net(model, loss_fn, X_train_inner_tensor, y_train_inner_tensor,n_replicates=1, max_iter=max_iter,tolerance=tolerance)
#             print('\n\tBest loss: {}\n'.format(best_final_loss))
            
#             # Make predictions 
#             y_test_est = best_net(X_test_inner_tensor)
    
#             # Determine errors and errors
#             se = (y_test_est.float()-y_test_inner_tensor.float())**2 # squared error
#             mse = (sum(se).type(torch.float)/len(y_test_inner_tensor)).data.numpy() #mean
            
#             test_error_ann[f,key] = mse
#         f = f + 1

#         # opt_val_err = np.min(MSEs_ann)
#         # opt_hidden_units = hidden_units[np.argmin(MSEs_ann)]
#         # print("Best loss: ", opt_val_err)
#         # print("Best hidden units: ", opt_hidden_units)
#         # k2 += 1
#     # --------------------------------------------------
#     # print(test_error_ann)
#     # print(best_final_loss)
#     opt_val_err_ann = np.min(np.mean(test_error_ann,axis=0))
#     opt_hidden_units = hidden_units[np.argmin(np.mean(test_error_ann,axis=0))]

#     print("Optimal error", opt_val_err_ann)
#     print("Optimal amount of hidden units", opt_hidden_units)

#     # Convert data to tensors 
#     X_train_tensor = torch.from_numpy(X_train_inner)
#     y_train_tensor = torch.from_numpy(y_train_inner)
#     X_test_tensor = torch.from_numpy(X_test_inner)
#     y_test_tensor = torch.from_numpy(y_test_inner)

#     # Convert tensors to float tensorst
#     X_train_tensor =  X_train_tensor.type(torch.FloatTensor)
#     y_train_tensor =  y_train_tensor.type(torch.FloatTensor)
#     X_test_tensor =  X_test_tensor.type(torch.FloatTensor)
#     y_test_tensor =  y_test_tensor.type(torch.FloatTensor)

#     # Create new model with optimal hidden units 
#     model = lambda: torch.nn.Sequential(
#                                     torch.nn.Linear(M, opt_hidden_units), #M features to n_hidden_units
#                                     torch.nn.Tanh(),   # 1st transfer function,
#                                     torch.nn.Linear(opt_hidden_units, 1), # n_hidden_units to 1 output neuron
#                                     # no final tranfer function, i.e. "linear output"
#                                     )

#     # Run ANN with this model
#     best_net, best_final_loss, best_learning_curve = train_neural_net(model, loss_fn, X_train_tensor, y_train_tensor,n_replicates=1, max_iter=max_iter,tolerance=tolerance)

#     y_test_est = best_net(X_test_tensor)
    
#     # Determine errors and errors
#     se = (y_test_est.float()-y_test_tensor.float())**2 # squared error
#     mse = (sum(se).type(torch.float)/len(y_test_tensor)).data.numpy() #mean
#     # errors.append(mse) # store error rate for current CV fold 
#     print(mse)
#     # Save test error and optimal hidden layers for ANN
#     opt_hidden_units_ann[k1] = opt_hidden_units
#     error_test_ann[k1] = mse
    
#     # Increment k-fold layer counter 
#     k1 += 1
# # ===========================================================================

# results_table[" "] = error_test_ann
# results_table["ANN"] = opt_hidden_units_ann

# print(results_table.to_latex()) 
