import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

# load the csv in a df
df = pd.read_csv('ab_test_data.csv')
print(df.head())
print(df.describe())

# get the sum of clicks for each group
print(df.groupby("group").sum("click"))

# Plotting the df using pyplot to get a clear visual comparison
plt.figure(figsize=(9, 6))
ax = sns.countplot(data=df, x="group", hue="click", palette="colorblind")
plt.title("Clicks distribution in both Experimental and Control groups")
plt.xlabel("Groups")
plt.ylabel("Count of clicks")
plt.legend(title="Click?", labels=["No", "Yes"])
plt.show()