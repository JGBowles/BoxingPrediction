import pandas as pd
from fancyimpute import KNN, MICE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_score, make_scorer, f1_score



df = pd.read_csv("bouts_out_new.csv")

# INITIAL INSPECTION

# This is the shape of the data
#print(df.shape)
print(df['result'].value_counts())
print(df.isnull().sum().sort_values(ascending=False))

# Remove draws
df = df[df.result != 'draw']
print("drawless shape")
print(df.shape)

# Random under sample
winAcount, winBcount = df.result.value_counts()
df_winA = df[df['result'] == "win_A"]
df_winB = df[df['result'] == "win_B"]
df_winA_reduced = df_winA.sample(winBcount)
df_winB_reduced = df_winB
# df_winA_reduced = df_winA.sample(5000, random_state=1)
# df_winB_reduced = df_winB.sample(5000, random_state=1)
df = pd.concat([df_winA_reduced, df_winB_reduced], axis=0)


# Shows that reach_b is missing 349K times, a lot of the score cards are 
# missing aswell as the physical features 
print(df.isnull().sum().sort_values(ascending=False))




# If I create a new dataframe with purely complete records I only get 2800 records
# removed_df = df.dropna(how='any')
# print("Inconsistent records removed shape")
# print(removed_df.shape)

# There's 3 unique values for a result meaning it is multiclass classification
print(df['result'].value_counts()) 

# PRE-PROCESSING AND CLEAN-UP


# Encode the label 
le = preprocessing.LabelEncoder().fit(df['result'])
encoded = le.transform(df['result'])
df['result'] = encoded
target = df['result']
clean_df = df.drop(['result'], axis=1) #trial 
print("Clean df shape " + str(clean_df.shape))

#print(clean_df.head)

#clean_df = clean_df.dropna(how='any')

# Models can only handle numeric features so I convert the non-numeric features
# into numeric using dummy features
clean_df = pd.get_dummies(clean_df)

# This results in more features 
#print("Clean dataframe columns")
#print(clean_df.columns)
#print(clean_df.shape)

# Convert result to numeric data 
#result_conversion = {'win_A': 0, 'win_B': 1, 'draw': 2}
#target = target.replace({'result': result_conversion}).infer_objects()
#print(type(target['result']))

#target = clean_df[['result_win_A', 'result_win_B', 'result_draw']]
#print("Target dummied")
#print(target.isnull().sum().sort_values(ascending=False))


# Imputes the mean for missing values - link to paper
# the_imputer = Imputer(missing_values= 'NaN', strategy='mean', axis=0)
# the_imputer.fit(clean_df)
# clean_df = pd.DataFrame(data=the_imputer.transform(clean_df), columns=clean_df.columns)
#clean_df = pd.DataFrame(KNN(k=3).complete(clean_df))
clean_df = pd.DataFrame(MICE().complete(clean_df))


# Chained Imputer




#clean_df = clean_df[clean_df.result != ]

# All records now have no missing features 
#print(clean_df.isnull().sum().sort_values(ascending=False))

# Test both imputed values aswell as a completely clean dataset

# SCALING 
# Use MinMaxScaler to scale all values
# USe for KNN algorithm 
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
scaled_df = scaler.fit_transform(clean_df)
clean_df = pd.DataFrame(scaled_df, columns=clean_df.columns)
# print("Type of scaled df " + str(type(clean_df)))
# print("Shape of scaled df " + str(clean_df.shape))

# Split the dataset, splits the dataset 75%/25%, shuffles the dataset (see the book)
X_train, X_test, y_train, y_test = train_test_split(
     clean_df, 
     target, test_size=0.5, random_state=0)



#print(y_test.shape)
#print(X_train.head)

# Split the reduced 2800 dataset, splits the dataset 75%/25%, shuffles the dataset



# Select the 20 best features to reduce dimensionality 
import sklearn.feature_selection


# create datasets with different Ks
selection = sklearn.feature_selection.SelectKBest(chi2, k=20)
selected_features = selection.fit(X_train, y_train) # on x_train and y_train but save cleandf_2 as usual 
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [clean_df.columns[i] for i in indices_selected]

clean_df2 = clean_df[colnames_selected]
print(colnames_selected)

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    clean_df2, 
    target, test_size=0.1, random_state=0)

# selection = sklearn.feature_selection.SelectKBest(chi2, k=20)
# selected_features = selection.fit(X_train, y_train)
# indices_selected = selected_features.get_support(indices=True)
# colnames_selected = [clean_df.columns[i] for i in indices_selected]

# X = clean_df[colnames_selected]

#print(clean_df2.shape)

# print(X_train_selected.columns)
# print(X_train_selected.head())




