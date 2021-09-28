import numpy as np
from numpy.lib.shape_base import expand_dims
import pandas as pd
from matplotlib import pyplot as plt
import load_data


df = load_data.df


# =============================================
# AGE - Barplot
# =============================================
age = df["age"]

# Count occurence of age 
age_unique, counts = np.unique(age, return_counts=True)

fig = plt.figure()
ax1 = fig.add_subplot()
ax1.bar(age_unique, counts)
ax1.set_xlabel("age [years]")
ax1.set_ylabel("count")
# plt.show()

# =============================================
# AGE and SEX - Barplot
# =============================================
age_sex = df[["age", "sex"]]

age_sex["sex"]
age_sex_unique, as_counts = np.unique(age_sex["sex"], return_counts=True)
# print(age_sex["sex"] == "Female")

# age_sex.pivot("column","sex","age").plot(kind="bar")
bar_width = 0.4

male_age = age_sex[age_sex["sex"] == "Male"]
female_age = age_sex[age_sex["sex"] == "Female"]

male_unique, male_counts = np.unique(male_age["age"], return_counts=True)
female_unique, female_counts = np.unique(female_age["age"], return_counts=True)
age_unique, counts = np.unique(age, return_counts=True)
