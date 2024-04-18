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