# Some code is reused from the Toolbox given in the course (02450)

from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl

import load_data

# Load the data 
df = load_data.df

# Set matplotlib style 
mpl.style.use("seaborn-colorblind")

# attributes to plot 
attribute_names = ["age", "capital-gain", "capital-loss", "hours-per-week"]
# Get the columns from the dataset and convert to numpy array
X = df[attribute_names].to_numpy()

# Class labels extraction
class_labels = df["annual-income"]
class_names = sorted(set(class_labels))
class_dict = dict(zip(class_names, range(2)))

# y is created 
y = np.asarray([class_dict[value] for value in class_labels])

# The values of N, M and C are encoded
N = len(y)
M = len(attribute_names)
C = len(class_names)

## Next we plot a number of atttributes
attributes = [0,1,2,3]
num_atr = len(attributes)


fig = plt.figure(figsize=(12,12))
for m1 in range(num_atr):
    for m2 in range(num_atr):
        plt.subplot(num_atr, num_atr, m1*num_atr + m2 + 1)
        for c in range(C):
            class_mask = (y==c)
            plt.plot(X[class_mask,attributes[m2]], X[class_mask,attributes[m1]], '.')
            if m1==num_atr-1:
                plt.xlabel(attribute_names[attributes[m2]])
            else:
                plt.xticks([])
            if m2==0:
                plt.ylabel(attribute_names[attributes[m1]])
            else:
                plt.yticks([])
            #ylim(0,X.max()*1.1)
            #xlim(0,X.max()*1.1)
plt.legend(class_names)
plt.savefig('plots/plot_matrix.jpg')

