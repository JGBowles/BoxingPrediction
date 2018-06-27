from sklearn.svm import SVC

#svc = SVC(kernel='linear', gamma=33, C=100)

# svc = SVC()

# scores = cross_val_score(svc, clean_df, target, cv=10, n_jobs=-1)
# print("Full features: mean of the scores: {:.2f}".format(scores.mean()))

# scores = cross_val_score(svc, clean_df2, target, cv=10)
# print("Reduced features: Mean of the scores: {:.2f}".format(scores.mean()))


#Random Grid Search
kernel = ['rbf', 'sigmoid', 'linear']
Cs = [x for x in np.linspace(start=0.01, stop=100, num=10)]
gammas = [x for x in np.linspace(start=0.01, stop=100, num=10)]
random_grid = {'kernel' : kernel, 'C' : Cs, 'gamma' : gammas}
svc = SVC()
svc_random = RandomizedSearchCV(estimator = svc, param_distributions = random_grid, n_iter = 100, cv = 3, n_jobs=-1, verbose=5, random_state=42)
svc_random.fit(X_train, y_train)
print(svc_random.best_params_)

# test each kernel, take best, then perform search 



# Grid search CV 

# param_grid = {
#     'max_depth' : [20, 25, 28, 30, 35],
#     'max_features': ['sqrt'],
#     'n_estimators' : [450, 460, 466, 470, 475]
# }

# forest = RandomForestClassifier()

# grid_search = GridSearchCV(estimator = forest, param_grid = param_grid, cv = 3, n_jobs=-1, verbose=3)
# grid_search.fit(X_train, y_train)

# print(grid_search.best_params_)


# grid_search = GridSearchCV(SVC(), param_grid, cv=3, n_jobs= -1, verbose=2)
# grid_search.fit(X_train, y_train)

# print(grid_search.best_params_)

# svc = SVC()
# clf = 
# svc.fit(X_train, y_train)
# print("Score " + str(svc.score(X_test, y_test)))
# Assess with cross validation
# Test on both the 20 features aswell as all features
