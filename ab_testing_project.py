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

# Calculate percentages and annotate each bar
total = df.groupby(["group"]).size()["exp"] # or len(df)/2 since there are 2 groups

for p in ax.patches:
    height = p.get_height()
    if height > 0: # We want to display the percentages only if the height of the bar is > 0
        percentage = '{:.1f}%'.format(100 * height / total)
        ax.text(p.get_x() + p.get_width() / 2., height + 40, percentage, ha="center", color='black')

plt.show()