import numpy as np 
import pandas as pd


data_path = "data/adult.data"
names_path = "data/adult.names"

# Open the names file and read lines 
with open(names_path) as f:
    # print(f.read())
    lines = f.readlines()

# Loop through lines and append column labels to list 
col_labels = []
for line in lines:
    if line[0] != "|" and ":" in line:
        # print(line)
        col_labels.append(line.split(":")[0])


print(col_labels)

df = pd.read_csv(data_path)

# print(df.head())

# print(df.columns)