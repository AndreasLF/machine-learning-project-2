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

# # =============================================
# # Write boxplots to file
# # =============================================

# Data to plot 
data_cols = ["age", "capital-gain", "capital-loss", "hours-per-week"]

# Loop through columns and make boxplot 
for label in data_cols:
    print("Creating box plot for label: " + label )
    fig = px.box(df, y=label, points="all")
    fig.write_image("plots/boxplot_"+ label +".jpg")


# =============================================
# Create histograms
# =============================================

# Age histogram colored by sex 
print("Creating histogram for: age")
fig = px.histogram(df, x = "age", color = "sex")
fig.update_layout(bargap = 0.01)
fig.write_image("plots/histogram_"+ "age" +".jpg")

# Loop through labels and create histogram plot 
for label in ["capital-gain", "capital-loss", "hours-per-week"]:
    print("Creating histogram for: " + label)
    fig = px.histogram(df, x = label)
    fig.update_layout(bargap = 0.01)
    fig.write_image("plots/histogram_"+ label +".jpg")
