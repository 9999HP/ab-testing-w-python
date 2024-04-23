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

# Variables that will be used for the computation of the pooled variance
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

# Pooled Variance
pooled_var = p_pooled * (1-p_pooled) * (1/N_control + 1/N_exp)
print(f"The pooled variance is equal to: {pooled_var}")

# Standard Error of the pooled variance
SE = np.sqrt(pooled_var)
print(f"Standard Error: {SE}")

# Z-test test statistics
stat_ztest = (p_control - p_exp) / SE
print(f"Test Statistics for 2-sample Z-test is: {stat_ztest}")

# critical value of the Z-test
z_crit = norm.ppf(1-alpha/2)
print(f"Z-critical value from Standard Normal distribution: {z_crit}")

# P-value calculation using the previous test statistics
p_value = 2 * norm.sf(abs(stat_ztest))
print(p_value)

# Function that checks the statistical significance
def statistical_significance(p_value, alpha):
  if p_value <= alpha:
    print("Since the p-value is less than the significance level, we can conclude that the observed differences between the groups are statistically significant and not due to random variation.")
  else:
    print("Since the p-value is greater than the significance level, we cannot conclude that the observed differences between the groups are statistically significant; they may be due to random variation.")

statistical_significance(p_value, alpha)

'''
Now that we know if there is a statistical significance, we must calculate the Confidence Interval in order to check for the practical significance
'''

# Calculate the CI (95%)
CI = [round((p_exp - p_control) - SE * z_crit, 4), round((p_exp - p_control) + SE * z_crit, 4)]

# Print the CI
print(f"The range for the confidence interval is: {CI}")

# Testing for practical significance using previous measures: Minimum Detectable Effect and the lower bound of the CI
def practically_significant(mde, lower_ci):
  if lower_ci >= mde:
    print(f"The MDE is {mde} and the minimum value of the CI is {lower_ci}")
    print("There is a practical significant difference between the Control and Experimental groups!")
  else:
    print(f"The MDE is {mde} and the minimum value of the CI is {lower_ci}")
    print("There is no practical significant difference between the Control and Experimental groups.")

practically_significant(mde, CI[0])