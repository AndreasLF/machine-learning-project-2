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

# =============================================
# Count NaN values 
# =============================================
print("Amount of NaN values")
print(df.isna().sum())

# =============================================
# Describe (statistcs) 
# =============================================
data_cols = ["age", "capital-gain", "capital-loss", "hours-per-week"]
# Take only data that are ratios or intevals 
sdf = df[data_cols]
print("Describe statistics")
print(sdf.describe())

# Print latex table 
print("Latex table: Describe statistics")
print(sdf.describe().to_latex())


# =============================================
# Male-female count
# =============================================
# Count males and females in dataset 
male_count = len(df[df["sex"] == " Male"])
female_count = len(df[df["sex"] == " Female"])

print("Male count: " + str(male_count))
print("Female count: " + str(female_count))
print("Male percentage: " + str(male_count/(male_count+female_count)*100))
print("Female percentage: " + str(female_count/(male_count+female_count)*100))

# =============================================
# Capital-gain/loss count
# =============================================
# Count males and females in dataset 
cap_gain_count = len(df[df["capital-gain"] == 0])
cap_loss_count = len(df[df["capital-loss"] == 0])

print("Capital gain = 0 count: " + str(cap_gain_count))
print("Capital gain = 0 percentage: " + str(cap_gain_count/len(df["capital-gain"])))

print("Capital gain = 0 count: " + str(cap_loss_count))
print("Capital gain = 0 percentage: " + str(cap_loss_count/len(df["capital-loss"])))

# =============================================
# Write boxplots to file
# =============================================

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
fig = px.histogram(df, x = "age", color = "sex", barmode="group")
fig.update_layout(bargap = 0.2)
fig.write_image("plots/histogram_"+ "age" +".jpg")

# Loop through labels and create histogram plot 
for label in ["capital-gain", "capital-loss", "hours-per-week"]:
    print("Creating histogram for: " + label)
    fig = px.histogram(df, x = label)
    fig.update_layout(bargap = 0.01)
    fig.write_image("plots/histogram_"+ label +".jpg")


# =============================================
# Create histograms
# =============================================
columns = list(df.columns)

nondata_columns = [x for x in columns if x not in data_cols]

# Loop through labels and create histogram plot 
for label in nondata_columns:
    print("Creating histogram for: " + label)
    fig = px.histogram(df, x = label, color=label)
    fig.update_layout(bargap = 0.01)
    fig.write_image("plots/histogram_"+ label +".jpg")
    