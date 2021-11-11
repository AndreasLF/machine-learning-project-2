import scipy.stats as st
import numpy as np
import pandas as pd
import scipy
from toolbox_02450 import correlated_ttest
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# Regression Data
regression_error = np.array([133.54,136.07,137.97,123.53,133.06,133.90,143.22,124.82,126.63,129.17])
baseline_error = np.array([152.60, 152.24, 151.59, 153.69, 152.11, 152.23, 151.36,153.10,153.26,152.33])
ann_error = np.array([118.06, 120.91, 125.94, 110.14, 119.70, 118.08, 128.86, 115.68, 114.20, 115.27])

# Classification Data
classification_tree_error = np.array([0.141278,0.144656,0.158170,0.146499,0.145885,0.145270,0.144349,0.146499,0.142506,0.141892])
classification_lreg_error = np.array([0.192568,0.186118,0.189803,0.199017,0.192875,0.194410,0.197482,0.197789,0.201474,0.189496])
classification_base_error = np.array([0.241982,0.241163,0.240480,0.240310,0.240208,0.240309,0.241436,0.240447,0.240344,0.241164])

# paired t-test values
alpha = 0.05
K=10
rho = 1/K

# Regression paired t-test
r = baseline_error - regression_error
p_baseline_regression, CI_baseline_regression = correlated_ttest(r, rho, alpha=alpha)


r = baseline_error - ann_error
p_baseline_ann, CI_baseline_ann = correlated_ttest(r, rho, alpha=alpha)


r = regression_error - ann_error
p_regression_ann, CI_regression_ann = correlated_ttest(r, rho, alpha=alpha)


# Classification paired t-test
r = classification_base_error - classification_lreg_error
p_baseline_lreg, CI_baseline_lreg = correlated_ttest(r, rho, alpha=alpha)

r = classification_base_error - classification_tree_error
p_baseline_tree, CI_baseline_tree = correlated_ttest(r, rho, alpha=alpha)

r = classification_lreg_error - classification_tree_error
p_lreg_tree, CI_lreg_tree = correlated_ttest(r, rho, alpha=alpha)

# Put data into numpy array and add conclusion (regression)
data = np.array([["E_baseline - E_regression", "E_baseline - E_ANN", "E_regression - E_ANN"],
        [CI_baseline_regression, CI_baseline_ann, CI_regression_ann], 
        [p_baseline_regression, p_baseline_ann, p_regression_ann],
        ["H_0 rejected,", "H_0 rejected", "H_0 rejected"]])

# Convert to dataframe 
df = pd.DataFrame(np.transpose(data),columns=["H_0", "Confidence interval", "p-value", "Conclusion"])
print(df)
# Convert to Latex 
print(df.to_latex())

# Put data into numpy array and add conclusion (classification)
data2 = np.array([["E_baseline - E_logistic_regression", "E_baseline - E_tree", "E_logistic_regression - E_tree"],
        [CI_baseline_lreg, CI_baseline_tree, CI_lreg_tree],
        [p_baseline_lreg, p_baseline_tree, p_lreg_tree],
        ["H_0 accepted,", "H_0 accepted", "H_0 accepted"]])

# Convert to dataframe
df2 = pd.DataFrame(np.transpose(data2),columns=["H_0", "Confidence interval", "p-value", "Conclusion"])
print(df2)
# Convert to Latex
print(df2.to_latex())
