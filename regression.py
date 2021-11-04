from transform_data import *
import sklearn.linear_model as lm

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

X = scaled_data

# =============================================
# Check if matrix is inversible 
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

plt.show()