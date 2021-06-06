import pandas as pd
from sklearn.model_selection import train_test_split

# import biscuit and cake datasets
df_biscuit = pd.read_csv("data/biscuit-recipes-complete.csv", header=0)
df_cake = pd.read_csv("data/cake-recipes-complete.csv", header=0)
# remove duplicate recipes before training
df_biscuit.drop_duplicates(inplace=True)  # 1479 values left
df_cake.drop_duplicates(inplace=True)  # 2803 values left

# get number of values in each dataset - results above
# print(df_cake.shape, df_biscuit.shape)

# concatenate them into one dataframe
df = pd.concat([df_cake, df_biscuit])
# get the list of recipe columns - these will be our features (once the label column is removed)
feature_cols = list(df.columns.values)
feature_cols.remove('label')
# splitting the dataframe into features and labels
X = df.loc[:, feature_cols]
y = df.label
# normalise rows of X (feature set) so that they sum to one to improve learning accuracy
X = X.div(X.sum(axis=1), axis=0)
# split into train and test sets with an 80:20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# import jaffa cake data
df_jaffa = pd.read_csv("data/jaffa-cake-recipes.csv")
# get the list of recipe columns - these will be our features (once the label column is removed)
feature_cols = list(df_jaffa.columns.values)
# splitting the dataframe into features and labels
X_jaffa = df_jaffa.loc[:, feature_cols]
X_jaffa = X_jaffa.div(X_jaffa.sum(axis=1), axis=0)
