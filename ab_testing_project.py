import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

'''
The A/B test dataset is a fictional dataset that I generated using Python.

I employed slightly different binomial distributions for the Control and Experimental groups to get varied results.
'''

# load the csv in a pandas df
df = pd.read_csv('ab_test_data.csv')
print(df.head())
print(df.describe())

# get the sum of clicks for each group (control and experimental groups)
print(df.groupby("group").sum("click"))

# Plotting the df using pyplot+seaborn to get a clear visual comparison
plt.figure(figsize=(9, 6))
ax = sns.countplot(data=df, x="group", hue="click", palette="colorblind")
plt.title("Clicks distribution in both Experimental and Control groups")
plt.xlabel("")
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

# Variables that will be used for the computation of the pooled variance in order to calculate the pooled SE later on
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

# Pooled Standard Error
SE = np.sqrt(pooled_var)
print(f"Standard Error: {SE}")

# 2-Sample Z-test test statistics calculation
stat_ztest = (p_control - p_exp) / SE
print(f"Test Statistics for 2-sample Z-test is: {stat_ztest}")

# Z-critical value calculation
z_crit = norm.ppf(1-alpha/2)
print(f"Z-critical value from Standard Normal distribution: {z_crit}")

'''
We already have important information about the significance of the results with these 2 values. Indeed, if the value of the test statistic is more extreme than the critical Z value, we can already say that there is a significant difference between groups and that we can reject the null hypothesis.

However, calculating the p-value remains a traditional step, so I'll also calculate it and compare the result with alpha (the significance level).
'''

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
Now that we know if there is a statistical significance, we can calculate the Confidence Interval in order to check for the practical significance.

If the lower bound of the CI is bigger than the MDE, we can say that there is a practical significance.

The width of the CI also gives us important information. The narrower it is, the more precise and generalizable the result is to the entire population.
'''

# Calculate the CI (95%)
CI = [round((p_exp - p_control) - SE * z_crit, 4), round((p_exp - p_control) + SE * z_crit, 4)]

# Print the CI
print(f"The range for the confidence interval is: {CI}")

# Testing for practical significance using previous measures: Minimum Detectable Effect and the lower bound of the CI
def practically_significant(mde, lower_ci):
  if lower_ci >= mde:
    print(f"The MDE is {mde} and the lower bound value of the CI is {lower_ci}")
    print("There is a practical significant difference between the groups!")
  else:
    print(f"The MDE is {mde} and the lower bound value of the CI is {lower_ci}")
    print("There is no practical significant difference between the groups.")

practically_significant(mde, CI[0])