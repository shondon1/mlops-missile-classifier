import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder




#This is Phase 1 
#TODO: Clean up comments
#loading training data
train_df = pd.read_csv("data/train.csv")
print(train_df.head())

#verifing data load
print("##########################")
# Check column types
print(train_df.shape)
print("##########################")

# Check for missing values
print(train_df.isnull().sum())
print("##########################")

# Get class distribution
print(train_df['reentry_phase'].value_counts())

# Plot histogram of altitude
train_df['altitude'].plot.hist(bins=50)
plt.show()