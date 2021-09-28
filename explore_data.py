import numpy as np
from numpy.lib.shape_base import expand_dims
import pandas as pd
from matplotlib import pyplot as plt
import load_data


df = load_data.df

# =============================================
# Count NaN values 
# =============================================
print("Amount of NaN values")
print(df.isna().sum())

# =============================================
# Describe (statistcs) 
# =============================================
print(df.describe())