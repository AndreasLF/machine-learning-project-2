import numpy as np
from numpy.lib.shape_base import expand_dims
import pandas as pd
from matplotlib import pyplot as plt
import load_data
import os

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if not os.path.exists("plots"):
    os.mkdir("plots")

df = load_data.df

# # =============================================
# # Count NaN values 
# # =============================================
print("Amount of NaN values")
print(df.isna().sum())

# # =============================================
# # Describe (statistcs) 
# # =============================================
data_cols = ["age", "capital-gain", "capital-loss", "hours-per-week"]
# Take only data that are ratios or intevals 
sdf = df[data_cols]
print("Describe statistics")
print(sdf.describe())

# Print latex table 
print("Latex table: Describe statistics")
print(sdf.describe().to_latex())
