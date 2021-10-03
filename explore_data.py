import numpy as np
from numpy.lib.shape_base import expand_dims
import pandas as pd
from matplotlib import pyplot as plt
import load_data
import os
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if not os.path.exists("plots"):
    os.mkdir("plots")

df = load_data.df

jpg_scale = 6

# =============================================
# Count NaN values 
# =============================================
print("Dataframe info")
print(df.info())

# =============================================
# Count NaN values 
# =============================================
print("Amount of NaN values")
print(df.isna().sum())
print(df.isna().sum().to_latex())


# =============================================
# Describe (statistcs) 
# =============================================
data_cols = ["age", "capital-gain", "capital-loss", "hours-per-week", "education-num"]
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
male_count = len(df[df["sex"] == "Male"])
female_count = len(df[df["sex"] == "Female"])

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
# Print attribute values (min/max if numerical)
# =============================================
cols = list(df.columns)

for col in cols:
    if type(df[col][0]) == str:
        print(col + ":")
        uniquelist = list(df[col].unique())
        uniquelist = map(str, uniquelist)
        print(", ".join(uniquelist))
    else:
        print(col + ":")
        print("Min: " + str(df[col].min()))
        print("Max: " + str(df[col].max()))
    print()

# =============================================
# Write boxplots to file
# =============================================

# Data to plot 
data_cols = ["age", "capital-gain", "capital-loss", "hours-per-week", "education-num"]

# Loop through columns and make boxplot 
for label in data_cols:
    print("Creating box plot for label: " + label )
    fig = px.box(df, y=label, points="all")
    fig.write_image("plots/boxplot_"+ label +".jpg", scale=jpg_scale)
    fig.write_image("plots/svg/boxplot_"+ label +".svg")



# =============================================
# Create histograms (numerical data)
# =============================================

# Age histogram colored by sex 
print("Creating histogram for: age")
fig = px.histogram(df, x = "age", color = "sex", barmode="group")
fig.update_layout(bargap = 0.2)
fig.write_image("plots/histogram_"+ "age_sex" +".jpg", scale=jpg_scale)

# Loop through labels and create histogram plot 
for label in ["capital-gain", "capital-loss", "hours-per-week", "education-num", "age"]:
    print("Creating histogram for: " + label)
    fig = px.histogram(df, x = label)
    fig.update_layout(bargap = 0.01)
    fig.write_image("plots/histogram_"+ label +".jpg", scale=jpg_scale)
    fig.write_image("plots/svg/histogram_"+ label +".svg")


# =============================================
# Create histograms
# =============================================
group_name = 'annual-income'

columns = list(df.columns)

nondata_columns = [x for x in columns if x not in data_cols]

# Loop through labels and create histogram plot 
for label in nondata_columns:

    if label != group_name:
        print("Creating histogram for: " + label)

        # Group by annual income
        df_group = df.groupby([label, group_name]).size().reset_index()
        # Add a percentage column 
        df_group['percentage'] = df.groupby([label, group_name]).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        # Set the colums in the new dataframe
        df_group.columns = [label, group_name, 'Counts', 'Percentage']
        # Make the barplot with calculated percentage 
        fig = px.bar(df_group, x=label, y=['Counts'], barmode="group", color=group_name, text=df_group['Percentage'].apply(lambda x: '{0:1.2f}%'.format(x)))
        # fig = px.histogram(df, x = label, color=label)
        # fig.update_layout(bargap = 0.01)
        fig.write_image("plots/histogram_percentages_"+ label +".jpg", scale=jpg_scale)
        fig.write_image("plots/svg/histogram_percentages_"+ label +".svg")

        fig = px.histogram(df, x = label)
        fig.update_layout(bargap = 0.01)
        fig.write_image("plots/histogram_"+ label +".jpg", scale=jpg_scale)
        fig.write_image("plots/svg/histogram_"+ label +".svg")



# =============================================
# Correlation matrix
# =============================================
# Define columns to include 
cols = ["age", "capital-gain", "capital-loss", "hours-per-week"] 
x= df[cols]

# Calculate correlation matrix 
corr = x.corr()

# Plot correlation matrix 
sns_heatmap = sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            annot=True, cmap="Blues")

fig = sns_heatmap.get_figure()

fig.savefig("plots/correlation_matrix.jpg", bbox_inches='tight', scale=jpg_scale)
fig.savefig("plots/svg/correlation_matrix.svg", bbox_inches='tight')

