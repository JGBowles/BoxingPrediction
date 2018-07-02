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
from sklearn.metrics import precision_score, make_scorer, f1_score, classification_report

df = pd.read_csv("bouts_out_new.csv")

# Remove draws
df = df[df.result != 'draw']
print("drawless shape")
print(df.shape)

# Random under sample - reduce to 82,000 records (roughly)
winAcount, winBcount = df.result.value_counts()
df_winA = df[df['result'] == "win_A"]
df_winB = df[df['result'] == "win_B"]
df_winA_reduced = df_winA.sample(winBcount)
df_winB_reduced = df_winB
df = pd.concat([df_winA_reduced, df_winB_reduced], axis=0)

# Encode the label 
le = preprocessing.LabelEncoder().fit(df['result'])
encoded = le.transform(df['result'])
df['result'] = encoded
target = df['result']
clean_df = df.drop(['result'], axis=1) #trial 
print("Clean df shape " + str(clean_df.shape))

# Models can only handle numeric features so I convert the non-numeric features - dummies
clean_df = pd.get_dummies(clean_df)

# Impute with MICE
clean_df = pd.DataFrame(MICE().complete(clean_df))


# SCALING 
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
scaled_df = scaler.fit_transform(clean_df)
clean_df = pd.DataFrame(scaled_df, columns=clean_df.columns)

# Split the dataset, splits the dataset 90/10%, shuffles the dataset (see the book)
X_train, X_test, y_train, y_test = train_test_split(
     clean_df, 
     target, test_size=0.1, random_state=0)


# Select the 20 best features to reduce dimensionality 
import sklearn.feature_selection
selection = sklearn.feature_selection.SelectKBest(chi2, k=20)
selected_features = selection.fit(X_train, y_train) 
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [clean_df.columns[i] for i in indices_selected]

# Do not need balanced data, purely for testing code
test_df = clean_df.sample(2500)
test_target = target.sample(2500)

x_train_test, x_test_test, y_train_test, y_test_test = train_test_split(
     test_df, 
     test_target, random_state=0)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def logreg_gridsearch(X, x, Y, y, param_grid):
    k = StratifiedKFold(n_splits=10, shuffle=False)
    logreg = LogisticRegression()
    logreg_grid = GridSearchCV(estimator = logreg, param_grid=param_grid, 
                            cv=k, n_jobs=-1, verbose=3)
    logreg_grid.fit(X, Y)
    prediction = logreg_grid.predict(x)
    print(classification_report(y, prediction))
    print(logreg_grid.best_estimator_.score(X, Y)) # 
    print(logreg_grid.best_estimator_.score(x, y))
    return logreg_grid.best_params_


# # Do a gridsearch on exponential values of 0.01 to 1000 for both sets of features
Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
param_grid = {'C':Cs}

# # # Full range of features 
print("Full range of features best parameters and results: ")
full_range_initial = logreg_gridsearch(X_train, X_test, y_train, y_test, param_grid)
print(full_range_initial)

# # # K20 range of features
print("K20 range of features best parameters: ")
k20_range_initial = logreg_gridsearch(X_train[colnames_selected], y_train, param_grid)
print(k20_range_initial)

# # Refined the param grid for full range
Cs = [x for x in range(50, 151)]
param_grid = {'C':Cs}
full_range_refine = logreg_gridsearch(X_train, X_test, y_train, y_test, param_grid)
print("Refined parameters results: ")
print(full_range_refine)

# # Refined the param grid for K20 range
Cs = [x for x in range(750, 1250)]
param_grid = {'C':Cs}
k20_range_refine = logreg_gridsearch(X_train[colnames_selected], X_test[colnames_selected], y_train, y_test, param_grid)
print("Refined parameters results - k20: ")
print(k20_range_refine)

Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# # Iterate through C levels - full range of features:
print("Full range of features - growth of C")
for c in Cs:
    logreg_clf = LogisticRegression(C=c)
    logreg_clf.fit(X_train, y_train)
    print("When the C is " + str(c) + " the score for the training set is " + str(logreg_clf.score(X_train, y_train)))
    print("When the C is " + str(c) + " the score for the test set is " + str(logreg_clf.score(X_test, y_test)))

print("K20 range of features - growth of C")
for c in Cs:
    logreg_clf = LogisticRegression(C=c)
    logreg_clf.fit(X_train[colnames_selected], y_train)
    print("When the C is " + str(c) + " the score for the training set is " + str(logreg_clf.score(X_train[colnames_selected], y_train)))
    print("When the C is " + str(c) + " the score for the test set is " + str(logreg_clf.score(X_test[colnames_selected], y_test)))


from sklearn.neighbors import KNeighborsClassifier

Ks = [x for x in range(1, 31)]
param_grid = {'n_neighbors': Ks}


def knn_gridsearch(X, x, Y, y, param_grid):
    k = StratifiedKFold(n_splits=10, shuffle=False)
    knn = KNeighborsClassifier()
    knn_grid = GridSearchCV(estimator = knn, param_grid=param_grid, 
                            cv=k, n_jobs=-1, verbose=3)
    knn_grid.fit(X, Y)
    prediction = knn_grid.predict(x)
    print(classification_report(y, prediction))
    print("Train set: ")
    print(knn_grid.best_estimator_.score(X, Y))
    print("Test set: ")
    print(knn_grid.best_estimator_.score(x, y))
    return knn_grid.best_params_

# Testing 1-30 for full range of features
print("Full range results:")
knn_initial = knn_gridsearch(X_train, X_test, y_train, y_test, param_grid)
print(knn_initial)

# Testing 1-30 for K20 range of features
print("K20 range results:")
knn_initial_k20 = knn_gridsearch(X_train[colnames_selected], X_test[colnames_selected], y_train, y_test, param_grid)
print(knn_initial_k20)

Ks = [1, 5, 10, 15, 20, 25, 30]

# Print growth of K and the accuracy on train/test set
print("Full range of features - growth of K")
for k in Ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    print("When K is " + str(k) + " the train score is " + str(knn.score(X_train, y_train)))
    print("When K is " + str(k) + " the test score is " + str(knn.score(X_test, y_test)))
    
print("K20 range of features - growth of K")
for k in Ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train[colnames_selected], y_train)
    print("When K is " + str(k) + " the train score is " + str(knn.score(X_train[colnames_selected], y_train)))
    print("When K is " + str(k) + " the test score is " + str(knn.score(X_test[colnames_selected], y_test)))
    
from sklearn.ensemble import RandomForestClassifier

def rdf_gridsearch(X, x, Y, y, param_grid):
    k = StratifiedKFold(n_splits=5, shuffle=False)
    rdf = RandomForestClassifier()
    rdf_grid = GridSearchCV(estimator = rdf, param_grid=param_grid,
                            cv=k, n_jobs=-1, verbose=51)
    rdf_grid.fit(X, Y)
    prediction = rdf_grid.predict(x)
    print(classification_report(y, prediction))
    print("Train set: ")
    print(rdf_grid.best_estimator_.score(X, Y))
    print("Test set: ")
    print(rdf_grid.best_estimator_.score(x, y))
    return rdf_grid.best_params_

n_estimators = [1, 20, 40, 60, 80, 100, 120, 128]
max_depth = [1, 10, 20, 30, 40, 50]
min_samples_leaf = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 200]

param_grid = {'n_estimators':n_estimators,
             'max_depth': max_depth,
             'min_samples_leaf':min_samples_leaf}

# # Initial test - full range
rdf_full_initial = rdf_gridsearch(X_train, X_test, y_train, y_test, param_grid)
print(rdf_full_initial)

# # K20 range
rdf_k20_initial = rdf_gridsearch(X_train[colnames_selected], X_test[colnames_selected], y_train, y_test, param_grid)
print(rdf_k20_initial)

# # Refined test - full range
n_estimators = [128]
max_depth = [57] # edit 
min_samples_leaf = [x for x in range(1, 100)]  

# #Further refined test
min_samples_leaf = [x for x in range(1, 100)]  
max_depth = [57] # edit 
param_grid = {'n_estimators':n_estimators,
             'max_depth': max_depth,
             'min_samples_leaf':min_samples_leaf}

rdf_full_refined = rdf_gridsearch(X_train, X_test, y_train, y_test, param_grid)
print(rdf_full_refined)

# #Refined K20 test
n_estimators = [120]
min_samples_leaf = [x for x in range(5, 15)]  
max_depth = [x for x in range(30, 50)]  
param_grid = {'n_estimators':n_estimators,
             'max_depth': max_depth,
             'min_samples_leaf':min_samples_leaf}

rdf_k20_refined = rdf_gridsearch(X_train[colnames_selected], X_test[colnames_selected], y_train, y_test, param_grid)
print(rdf_k20_refined)
min_samples_leaf = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


print("Full range of features, change in min samples")
for mini in min_samples_leaf:
    rdf_full = RandomForestClassifier(min_samples_leaf=mini)
    rdf_full.fit(X_train, y_train)
    print("When min is " + str(mini) + "Training set score: ")
    print(rdf_full.score(X_train, y_train))
    print("When min is " + str(mini) + "Ttest set score: ")
    print(rdf_full.score(X_test, y_test))

# # print("K20 range of features, change in min samples")
for min in min_samples_leaf:
    rdf_k20 = RandomForestClassifier(min_samples_leaf=min)
    rdf_k20.fit(X_train[colnames_selected], y_train)
    print("When min is " + str(min) + "Training set score: ")
    print(rdf_k20.score(X_train[colnames_selected], y_train))
    print("When min is " + str(min) + "Ttest set score: ")
    print(rdf_k20.score(X_test[colnames_selected], y_test))  
    
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB



def bn_gridsearch(X, x, Y, y, param_grid):
    k = StratifiedKFold(n_splits=10, shuffle=False)
    bn = BernoulliNB()
    bn_grid = GridSearchCV(estimator = bn, param_grid=param_grid,
                            cv=k, n_jobs=-1, verbose=51)
    bn_grid.fit(X, Y)
    prediction = bn_grid.predict(x)
    print(classification_report(y, prediction))
    print("Train set: ")
    print(bn_grid.best_estimator_.score(X, Y))
    print("Test set: ")
    print(bn_grid.best_estimator_.score(x, y))
    return bn_grid.best_params_

alphas = [0.001, 0.01, 0.1, 1, 10, 100]
param_grid = {'alpha':alphas}

print("Initial search - full range")
bn_initial = bn_gridsearch(X_train, X_test, y_train, y_test, param_grid)
print(bn_initial)

alphas = [x for x in np.linspace(0.0001, 0.1)]
param_grid = {'alpha':alphas}

print("Refined search - full range")
bn_initial = bn_gridsearch(X_train, X_test, y_train, y_test, param_grid)
print(bn_initial)

alphas = [0.001, 0.01, 0.1, 1, 10, 100]
param_grid = {'alpha':alphas}

print("Initial search - K20 range")
bn_initial = bn_gridsearch(X_train[colnames_selected], X_test[colnames_selected], y_train, y_test, param_grid)
print(bn_initial)

alphas = [0.001, 0.01, 0.1, 1, 10, 100]
param_grid = {'alpha':alphas}

for a in alphas:
    bn = BernoulliNB(alpha=a)
    bn.fit(X_train[colnames_selected], y_train)
    print("When the alpha is " + str(a) + " the train score is " + str(bn.score(X_train[colnames_selected], y_train)))
    print("When the alpha is " + str(a) + " the test score is " + str(bn.score(X_test[colnames_selected], y_test)))


from sklearn.neural_network import MLPClassifier
import itertools 

# Do gridsearch on parameters - both sets of features 
# Print most important parameter change variation - both sets of features 

def mlp_gridsearch(X, x, Y, y, param_grid):
    k = StratifiedKFold(n_splits=10, shuffle=False)
    mlp = MLPClassifier()
    mlp_grid = GridSearchCV(estimator = mlp, param_grid=param_grid, 
                            cv=k, n_jobs=-1, verbose=51)
    mlp_grid.fit(X, Y)
    prediction = mlp_grid.predict(x)
    print(classification_report(y, prediction))
    print("Train set: ")
    print(mlp_grid.best_estimator_.score(X, Y))
    print("Test set: ")
    print(mlp_grid.best_estimator_.score(x, y))
    return mlp_grid.best_params_


#hidden_layer_sizes = ([x for x in itertools.product((10, 25, 50, 53, 73, 75, 100, 125, 150, 175, 200), repeat=1)] + \
                      #[x for x in itertools.product((10, 25, 50, 53, 73, 75, 100, 125, 150, 175, 200), repeat=2)] )

hidden_layer_sizes = ([x for x in itertools.product((10, 50, 53, 73, 75, 125, 200), repeat=1)] + \
                      [x for x in itertools.product((10, 50, 53, 73, 75, 125, 200), repeat=2)] )
    
alpha = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
activation = ['relu', 'logistic', 'tanh']  
solver=['adam']

param_grid = {'hidden_layer_sizes':hidden_layer_sizes,
             'alpha': alpha,
             'activation':activation,
             'solver':solver}

#MLP initial search
print("MLP full range initial search")
full_mlp_initial = mlp_gridsearch(X_train, X_test, y_train, y_test, param_grid)
print(full_mlp_initial)

K20 initial search 
print("MLP K20 range initial search")
k20_mlp_initial = mlp_gridsearch(X_train[colnames_selected], X_test[colnames_selected], y_train, y_test, param_grid)
print(k20_mlp_initial)

Refined search for full range 
hidden_layer_sizes = ([x for x in itertools.product((x for x in range(190, 211)), repeat=1)])
alpha = [0.00001]
activation = ['relu', 'logistic', 'tanh'] 
solver=['adam']

 param_grid = {'hidden_layer_sizes':hidden_layer_sizes,
              'alpha': alpha,
              'activation':activation,
              'solver':solver}

print("MLP full range refined search")
full_mlp_refined = mlp_gridsearch(X_train, X_test, y_train, y_test, param_grid)
print(full_mlp_refined)

# Refined search for K20 range
list1 = []
for x in range(190, 211):
     list1.append(x)

print(list1)

list2 = []
for x in range(5, 16):
     list2.append(x)

print(list2)

hidden_layer_sizes=[]
for combo in itertools.product(list2, list1):
     hidden_layer_sizes.append(combo)
    
alpha = [0.00001]
activation = ['relu', 'logistic', 'tanh'] 
solver=['adam']

param_grid = {'hidden_layer_sizes':hidden_layer_sizes,
              'alpha': alpha,
              'activation':activation,
              'solver':solver}

print("MLP K20 range refined search")
k20_mlp_refined = mlp_gridsearch(X_train[colnames_selected], X_test[colnames_selected], y_train, y_test, param_grid)
print(k20_mlp_refined)

# 1 layer full range + k20 range - /train and test 
hidden_layer_sizes = [(10,), (25,), (50,), (75,), (100,), (125,), (150,), (175,), (200,)]
alpha = [0.00001]
activation = ['relu', 'logistic', 'tanh'] 
solver=['adam']

print("Full range of features - 1 layer: ")
for layer in hidden_layer_sizes:
    mlp_full=MLPClassifier(hidden_layer_sizes=layer)
    mlp_full.fit(X_train, y_train)
    print("For layer = " + str(layer) + "train = " + str(mlp_full.score(X_train, y_train)))
    print("For layer = " + str(layer) + "test = " + str(mlp_full.score(X_test, y_test)))

# # 1 layer full range + k20 range - train and test
print("K20 range of features - 1 layer: ")
for layer in hidden_layer_sizes:
    mlp_full=MLPClassifier(hidden_layer_sizes=layer)
    mlp_full.fit(X_train[colnames_selected], y_train)
    print("For layer = " + str(layer) + "train = " + str(mlp_full.score(X_train[colnames_selected], y_train)))
    print("For layer = " + str(layer) + "test = " + str(mlp_full.score(X_test[colnames_selected], y_test)))
    
# 2 layer full range + full/k20 range - /train and test 
hidden_layer_sizes = [(10,10), (25,25), (50,50), (75,75), (100,100), (125,125), (150,150), (175,175), (200,200)]
alpha = [0.00001]
activation = ['relu', 'logistic', 'tanh']  
solver=['adam']

print("Full range of features - 2 layer: ")
for layer in hidden_layer_sizes:
    mlp_full=MLPClassifier(hidden_layer_sizes=layer)
    mlp_full.fit(X_train, y_train)
    print("For layer = " + str(layer) + "train = " + str(mlp_full.score(X_train, y_train)))
    print("For layer = " + str(layer) + "test = " + str(mlp_full.score(X_test, y_test)))

# 2 layer full range + k20 range - train and test
print("K20 range of features - 2 layer: ")
for layer in hidden_layer_sizes:
    mlp_full=MLPClassifier(hidden_layer_sizes=layer)
    mlp_full.fit(X_train[colnames_selected], y_train)
    print("For layer = " + str(layer) + "train = " + str(mlp_full.score(X_train[colnames_selected], y_train)))
    print("For layer = " + str(layer) + "test = " + str(mlp_full.score(X_test[colnames_selected], y_test)))
    

hidden_layer_sizes = ([x for x in itertools.product((10, 25, 50, 53, 73, 75, 100, 125, 200), repeat=1)] + \
                      [x for x in itertools.product((10, 25, 50, 53, 73, 75, 100, 125, 200), repeat=2)] )
    
alpha = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
activation = ['relu', 'logistic', 'tanh'] 
solver=['adam']

full_mlp_refined = mlp_gridsearch(x_train_test, x_test_test, y_train_test, y_test_test, param_grid)
print(full_mlp_refined)

mlp = MLPClassifier(hidden_layer_sizes = (100,))

scores = cross_val_score(mlp, clean_df, target, cv=10, n_jobs=-1)
print(scores.mean())


from sklearn.svm import SVC, LinearSVC # remove l9inear

# Do gridsearch on parameters - both sets of features 
# Retrieve best parameters, refine parameters from selection
# Print most important parameter change variation - both sets of features 

def svc_gridsearch(X, x, Y, y, param_grid):
    k=StratifiedKFold(n_splits=10, shuffle=False)
    svc = SVC()
    svc_grid = GridSearchCV(estimator = svc, param_grid=param_grid, cv=k, n_jobs=-1,
                           verbose=51)
    svc_grid.fit(x, y)
    prediction = svc_grid.predict(x)
    print(classification_report(y, prediction))
    print("Train set: ")
    print(svc_grid.best_estimator_.score(X, Y))
    print("Test set: ")
    print(svc_grid.best_estimator_.score(x, y))
    return svc_grid.best_params_

kernel = ['linear', 'rbf'] 
Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
gammas = [0.001, 0.01, 0.1, 1, 10, 100, 1000] 
param_grid= {'C': Cs, 'gamma': gammas, 'kernel': kernel}
    
# Full range of features initial test
print("Full range of features:")
full_initial_svc = svc_gridsearch(X_train, x_test, Y_train, y_test)

# K20 range of features initial test
print("K20 range of features:")
k20_initial_svc = svc_gridsearch(X_train[colnames_selected}, x_test[colnames_selected], Y_train, y_test)

def linearsvc_gridsearch(X, x, Y, y, param_grid):
    k=StratifiedKFold(n_splits=10, shuffle=False)
    svc = LinearSVC()
    svc_grid = GridSearchCV(estimator = svc, param_grid=param_grid, cv=k, n_jobs=-1,
                           verbose=51)
    svc_grid.fit(x, y)
    prediction = svc_grid.predict(x)
    print(classification_report(y, prediction))
    print("Train set: ")
    print(svc_grid.best_estimator_.score(X, Y))
    print("Test set: ")
    print(svc_grid.best_estimator_.score(x, y))
    return svc_grid.best_params_

Cs = [x for x in range(30, 125)]
param_grid= {'C': Cs}
#Full range of features refined linear test
print("Full range of features:")
full_refined_svc = linearsvc_gridsearch(X_train, X_test, y_train, y_test, param_grid)
print(full_refined_svc)

#K20 range of features refined linear test
print("K20 range of features:")
k20_refined_svc = linearsvc_gridsearch(X_train[colnames_selected], X_test[colnames_selected], y_train, y_test, param_grid)
print(k20_refined_svc)

gammas = [0.001, 0.01, 0.1, 1, 10, 100, 1000] 

print("Analysing gamma change with the RBF kernel - full features")
for g in gammas:
    svc = SVC(kernel='rbf', gamma=g)
    svc.fit(X_train, y_train)
    print("When gamma = " + str(g) + "the train score = " + str(svc.score(X_train, y_train)))
    print("When gamma = " + str(g) + "the test score = " + str(svc.score(X_test, y_test)))

print("Analysing gamma change with the RBF kernel - K20 features")
for g in gammas:
    svc = SVC(kernel='rbf', gamma=g)
    svc.fit(X_train[colnames_selected], y_train)
    print("When gamma = " + str(g) + "the train score = " + str(svc.score(X_train[colnames_selected], y_train)))
    print("When gamma = " + str(g) + "the test score = " + str(svc.score(X_test[colnames_selected], y_test)))
    
svc = SVC(kernel='rbf', gamma=1000)
svc.fit(X_train[colnames_selected], y_train)
print("When gamma = " + str(g) + "the train score = " + str(svc.score(X_train[colnames_selected], y_train)))
print("When gamma = " + str(g) + "the test score = " + str(svc.score(X_test[colnames_selected], y_test)))
        
                                         
                                









