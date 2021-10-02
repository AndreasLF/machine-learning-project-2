import numpy as np 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

data_path = "data/adult.data"
names_path = "data/adult.names"

# Open the names file and read lines 
with open(names_path) as f:
    # print(f.read())
    lines = f.readlines()

# Loop through lines and append column labels to list 
col_labels = []
for line in lines:
    line.rstrip().replace(', ', '|||').replace(',', '```').replace('|||', ', ').replace('```', '|')
    if line[0] != "|" and ":" in line:
        # print(line)
        col_labels.append(line.split(":")[0])


# Add the label colum name 
col_labels.append("annual-income")

# Read the data file 
df = pd.read_csv(data_path, delimiter=", ")

# Set the column labels 
df.columns = col_labels

# Replace '?' with NaN values
df[df == '?'] = np.nan


workclass_columns = df["workclass"].unique()
print(workclass_columns)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
one_hot_encoding = pd.get_dummies(workclass_columns, prefix="ENCODED")
# print(one_hot_encoding)
