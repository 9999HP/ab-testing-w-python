import pandas as pd
import numpy as np

N_experimental = 10000
N_control = 10000

# Creation of 2 pandas series simulating click data on the population
clicks_experimental = pd.Series(np.random.binomial(1, 0.4, size=N_experimental))
clicks_control = pd.Series(np.random.binomial(1, 0.2, size=N_control))

# Converting series in df + adding identification
df_exp = pd.DataFrame(clicks_experimental, columns=["click"])
df_exp.insert(1, "group", "exp")

df_control = pd.DataFrame(clicks_control, columns=["click"])
df_control.insert(1, "group", "control")

# Concatenation of both df in a single one
df = pd.concat([df_exp, df_control], axis=0).reset_index(drop=True)
df.insert(0, "user_id", pd.RangeIndex(start=1, stop=len(df) + 1))
print(df.head(), df.tail())

# Check if the user_id isn't restarting when switching from exp to control
print(df.groupby("group").max("user_id"))

# Export to a CSV file
df.to_csv("ab_test_data.csv", index=False)