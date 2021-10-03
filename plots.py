import numpy as np
from numpy.lib.shape_base import expand_dims
import pandas as pd
from matplotlib import pyplot as plt
import load_data

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from plotly.subplots import make_subplots
import statsmodels.api as sm

from scipy import stats

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


# =============================================
# Create Distributions
# =============================================
age = df["age"]
mean = np.mean(age)
sd = np.std(age)
print("Mean age: ", mean)
print("Standard devation age: ", sd)
#create histogram
f = plt.figure()
plt.title('Age Histogram with Normal Distribution')
plt.hist(age, bins=74, density=True)
#create normal dist red line
ages = np.linspace(age.min(), age.max(), 1000)
pdf = stats.norm.pdf(ages,loc=mean,scale=sd)
plt.plot(ages,pdf,'.',color='red')
#create mean vertical line
min_ylim, max_ylim = plt.ylim()
plt.axvline(mean, color='k', linestyle='dashed', linewidth=1)
plt.text(mean*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(mean))

plt.savefig("plots/histogram_"+ "age_normal3" +".jpg")


print("Creating histogram for: age with normal distribution")
fig = ff.create_distplot([age], group_labels=['age'], curve_type='normal', colors=['blue'])
fig.update_layout(title_text='Distplot with Normal Distribution')
fig.write_image("plots/histogram_"+ "age_normal" +".jpg")

fig = go.FigureWidget(ff.create_distplot([age], group_labels=['age'], show_hist=True))
fig.layout.update(title='Density curve')
fig.write_image("plots/histogram_"+ "age_normal2" +".jpg")

#For hours per week
hours = df["hours-per-week"]
h_mean = np.mean(hours)
h_sd = np.std(hours)
print("Mean hours: ", h_mean)
print("Standard devation hours: ", h_sd)
#create histogram
f = plt.figure()
plt.title('Hours per Week Histogram with Normal Distribution')
plt.hist(hours, bins=99, density=True)
#create normal dist red line
hoursN = np.linspace(hours.min(), hours.max(), 1000)
pdf = stats.norm.pdf(hoursN,loc=h_mean,scale=h_sd)
plt.plot(hoursN,pdf,'.',color='red')
#create mean vertical line
min_ylim, max_ylim = plt.ylim()
plt.axvline(h_mean, color='k', linestyle='dashed', linewidth=1)
plt.text(h_mean*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(h_mean))

plt.savefig("plots/histogram_"+ "hours_normal" +".jpg")

#To check normal distribution we use QQ plot
#with standardized line
age = df['age']
fig = sm.qqplot(age, line ='s')
plt.savefig("plots/qqplot_"+ "age_normalS" +".jpg")

fig = sm.qqplot(hours, line ='s')
plt.savefig("plots/qqplot_"+ "hours_normalS" +".jpg")


