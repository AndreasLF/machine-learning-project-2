# Some code is reused from the Toolbox given in the course (02450)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import load_data
from sklearn import tree
from platform import system
from os import getcwd
from toolbox_02450 import windows_graphviz_call
import matplotlib.pyplot as plt
from matplotlib.image import imread

df = load_data.df

path_to_graphviz = r'C:\Program Files\Graphviz'

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


# Fit regression tree classifier, Gini split criterion, no pruning
criterion = 'gini'
dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=2)
dtc = dtc.fit(X, y)

fname = 'tree_' + criterion
# Export tree graph .gvz file to parse to graphviz
out = tree.export_graphviz(dtc, out_file='plots2\\'+ fname + '.gvz', feature_names=attribute_names)

# Depending on the platform, we handle the file differently, first for Linux
# Mac
if system() == 'Linux' or system() == 'Darwin':
    import graphviz

    # Make a graphviz object from the file
    src = graphviz.Source.from_file(fname + '.gvz')
    print('\n\n\n To view the tree, write "src" in the command prompt \n\n\n')

# ... and then for Windows:
if system() == 'Windows':
    # N.B.: you have to update the path_to_graphviz to reflect the position you 
    # unzipped the software in!
    windows_graphviz_call(fname=fname,
                          cur_dir=getcwd(),
                          path_to_graphviz=path_to_graphviz)

    plt.figure(figsize=(12, 12))
    plt.imshow(imread('plots2/' + fname + '.png'))
    plt.box('off');
    plt.axis('off')
    plt.show()

