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

# Plotting the df using pyplot+seaborn to get a clear visual comparison
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
plt.close()

# Significance Level and Minimum Detectable Effect (delta or mde) in %
alpha = 0.05
mde = 0.10

N_control = len(df[df["group"] == "control"])
N_exp = len(df[df["group"] == "exp"])
X_control = df[df["group"] == "control"]["click"].sum()
X_exp = df[df["group"] == "exp"]["click"].sum() 

# Click probability estimates per group
p_control = X_control/N_control
p_exp = X_exp/N_exp

# Pooled clicked probability
p_pooled = (X_control + X_exp)/(N_control + N_exp)

print(f"Click probability in the control group: {p_control}")
print(f"Click probability in the experimental group: {p_exp}")
print(f"Pooled Click probability: {p_pooled}")

# Continue with Pooled Variance, Standard Error and T Test