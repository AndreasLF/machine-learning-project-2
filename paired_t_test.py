import scipy.stats as st
import numpy as np
import pandas as pd
import scipy
from toolbox_02450 import correlated_ttest



regression_error = np.array([133.54,136.07,137.97,123.53,133.06,133.90,143.22,124.82,126.63,129.17])
baseline_error = np.array([152.60, 152.24, 151.59, 153.69, 152.11, 152.23, 151.36,153.10,153.26,152.33])
ann_error = np.array([118.06, 120.91, 125.94, 110.14, 119.70, 118.08, 128.86, 115.68, 114.20, 115.27])

alpha = 0.05
K=10
rho = 1/K

r = baseline_error - regression_error
p_baseline_regression, CI_baseline_regression = correlated_ttest(r, rho, alpha=alpha)


r = baseline_error - ann_error
p_baseline_ann, CI_baseline_ann = correlated_ttest(r, rho, alpha=alpha)


r = regression_error - ann_error
p_regression_ann, CI_regression_ann = correlated_ttest(r, rho, alpha=alpha)


# Put data into numpy array and add conclusion 
data = np.array([["E_baseline - E_regression", "E_baseline - E_ANN", "E_regression - E_ANN"],
        [CI_baseline_regression, CI_baseline_ann, CI_regression_ann], 
        [p_baseline_regression, p_baseline_ann, p_regression_ann],
        ["H_0 rejected,", "H_0 rejected", "H_0 rejected"]])

# Convert to dataframe 
df = pd.DataFrame(np.transpose(data),columns=["H_0", "Confidence interval", "p-value", "Conclusion"])
print(df)
# Convert to Latex 
print(df.to_latex())


