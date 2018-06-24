import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import chi2

df = pd.read_csv("bouts_out_new.csv")

# INITIAL INSPECTION

# This is the shape of the data
print(df.shape)
print(df['result'].value_counts())

# Remove draws
df = df[df.result != 'draw']
print("drawless shape")
print(df.shape)

# Random under sample
winAcount, winBcount = df.result.value_counts()
df_winA = df[df['result'] == "win_A"]
df_winB = df[df['result'] == "win_B"]
df_winA_reduced = df_winA.sample(winBcount)
df = pd.concat([df_winA_reduced, df_winB], axis=0)
print(df.result.value_counts())
print(df.shape)


# These are the features I am working with
#print(df.columns)

# Shows that reach_b is missing 349K times, a lot of the score cards are 
# missing aswell as the physical features 
#print(df.isnull().sum().sort_values(ascending=False))

# The data shows that fighter A is recorded as winning the most 


# If I create a new dataframe with purely complete records I only get 2800 records
# removed_df = df.dropna(how='any')
# print("Inconsistent records removed shape")
# print(removed_df.shape)

# There's 3 unique values for a result meaning it is multiclass classification
#print(df['result'].value_counts()) 

# PRE-PROCESSING AND CLEAN-UP

# Encode the label 
le = preprocessing.LabelEncoder().fit(df['result'])
encoded = le.transform(df['result'])
df['result'] = encoded
target = df['result']
clean_df = df.drop(['result'], axis=1)


#print(clean_df.head)

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
the_imputer = Imputer(missing_values= 'NaN', strategy='mean', axis=0)
the_imputer.fit(clean_df)
clean_df = pd.DataFrame(data=the_imputer.transform(clean_df), columns=clean_df.columns)


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
#print("Type of scaled df " + str(type(clean_df)))
#print("Shape of scaled df " + str(clean_df.shape))


# Split the dataset, splits the dataset 75%/25%, shuffles the dataset (see the book)
X_train, X_test, y_train, y_test = train_test_split(
     clean_df, 
     target, random_state=0)



#print(y_test.shape)
#print(X_train.head)

# Split the reduced 2800 dataset, splits the dataset 75%/25%, shuffles the dataset
X_train, X_test, y_train, y_test = train_test_split(
    clean_df, 
    target, random_state=0)


# Select the 20 best features to reduce dimensionality 
import sklearn.feature_selection

selection = sklearn.feature_selection.SelectKBest(chi2, k=20)
selected_features = selection.fit(clean_df, target)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [clean_df.columns[i] for i in indices_selected]

clean_df2 = clean_df[colnames_selected]

#print(clean_df2.shape)

# print(X_train_selected.columns)
# print(X_train_selected.head())

