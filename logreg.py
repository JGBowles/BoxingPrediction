from sklearn.linear_model import LogisticRegression



# logreg = LogisticRegression(random_state=0)
# Cs = [x for x in np.linspace(start=1, stop=100, num=10)] 
# param_grid = { 'C':Cs }
# prec_metric = make_scorer(precision_score)
# k = StratifiedKFold(n_splits=10, shuffle=False)
# k2 = StratifiedKFold(n_splits=10, shuffle=False)
# grid = RandomizedSearchCV(estimator = logreg, param_distributions=param_grid, 
#                     scoring=prec_metric,cv=k, 
#                     n_jobs=-1, verbose=3)
# grid.fit(X_train, y_train)
# print("Grid search 1 best params " + str(grid.best_params_))

# # Get score of X_train, y_train (TEST FOLD) from GridSearchCV metric
# print("Mean test score from GridSearchCV: " + (str(grid.cv_results_['mean_test_score'].mean())))

# # Feed result of GridSearchCV to cross_val_score for scoring on the test set, is it similar to above score?
# scores = cross_val_score(grid.best_estimator_, X_test, 
#                          y_test,cv=k)

# print("Scores on hold out set with cross_val_score: " + str(scores.mean()))


# logreg2 = LogisticRegression()

# prec_metric = make_scorer(precision_score)

# grid2 = RandomizedSearchCV(estimator = logreg2, param_distributions=param_grid, cv=k2, 
#                     n_jobs=-1, scoring=prec_metric, verbose=3)
# grid2.fit(clean_df, target)
# print("Grid search 1 best params " + str(grid2.best_params_))
# # Score of the whole dataset on the test fold with GridSearchCV 
# print("Mean test score from GridSearchCV full dataset: " + (str(grid2.cv_results_['mean_test_score'].mean())))




# scores2 = cross_val_score(grid.best_estimator_, clean_df, 
#                          target,cv=10)

print("Scores on hold out set with cross_val_score: " + str(scores2.mean()))









# scores2 = cross_val_score(grid.best_estimators_, X_test2, y_test)

# print("Dataset of 36: Mean of the scores: {:.2f}".format(scores.mean()))



# logreg = LogisticRegression()

# grid_search = GridSearchCV(estimator = logreg, param_grid=param_grid, cv=3, n_jobs=-1, verbose=3)
# grid_search.fit(clean_df, target)

# print(grid_search.cv_results_)


# Cs = [0.01, 0.1, 1, 10, 100]

# for c in Cs:
#     logreg = LogisticRegression(C=c)
#     logreg.fit(X_train, y_train)
#     scores = cross_val_score(logreg, X_test, y_test, cv=10)
#     print("Dataset with 36 features scores: {}".format(scores))
#     print("Mean of the scores: {:.2f}".format(scores.mean()))

#     scores = cross_val_score(logreg, X_test, y_test, cv=10)
#     print("Dataset with 20 features scores: {}".format(scores))
#     print("Mean of the scores: {:.2f}".format(scores.mean()))




# logreg = LogisticRegression()

# Cs = [x for x in np.linspace(start=0.01, stop=100)]
# param_grid = { 'C':Cs }

# grid = GridSearchCV(estimator = logreg, param_grid=param_grid, cv=3, n_jobs=-3, verbose=3)
# grid.fit(X_train2, y_train2)

# scores = cross_val_score(grid.best_estimator_, X_test2, y_test2, cv=10)
# print("Dataset of 20: Mean of the scores: {:.2f}".format(scores.mean()))











#

# logreg = LogisticRegression(C=75)
# logreg.fit(X_train, y_train)





# logreg = LogisticRegression()



# # print(grid_search.best_params_)

# logreg = LogisticRegression(C=84)

# print(logreg.score(X_test, y_test))

# logreg2 = LogisticRegression(C=10)
# scores = cross_val_score(logreg2, clean_df, target, cv=10)
# print(scores.mean())
# print(logreg.score(clean_df, target))







# Grid search CV 

# 

# param_grid = {
#     'C': Cs
# }

# logreg = LogisticRegression()
# grid_search = GridSearchCV(estimator = logreg, param_grid = param_grid, cv = 3, n_jobs=-1, verbose=3)
# grid_search.fit(X_train, y_train)
# print(grid_search.best_params_)

# Second manual grid search

# Cs = [0.01, 0.1, 1, 10, 100]

# for c in Cs:
#     logreg = LogisticRegression(C=c)
#     scores = cross_val_score(logreg, clean_df, target, cv=10)
#     print("Full features: When C = " + str(c) + ". Mean of the scores: {:.2f}".format(scores.mean()))
#     scores = cross_val_score(logreg, clean_df2, target, cv=10)
#     print("Reduced features: When C = " + str(c) + ". Mean of the scores: {:.2f}".format(scores.mean()))


# X_train, X_test, y_train, y_test = train_test_split(
#       clean_df, 
#       target, random_state=0)

# logreg = LogisticRegression().fit(X_train, y_train)
# print(logreg.score(X_train, y_train))
# print(logreg.score(X_test, y_test))


# High training set accuracy/low test set accuracy means overfitting 
# When both train/test is similar, means underfitting 

#Random Grid Search
# Cs = [x for x in np.linspace(start=0.001, stop=1000, num=10 )]

# random_grid = {'C': Cs}

# logreg = LogisticRegression()

# lr_random = RandomizedSearchCV(estimator = logreg, param_distributions = random_grid, n_iter = 10, cv = 3, n_jobs=-1, verbose=3, random_state=42)
# lr_random.fit(X_train, y_train)
# print(lr_random.best_params_)

# logreg = LogisticRegression(C=1).fit(X_train, y_train)
# print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
# print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))


# C = [0.01, 0.1, 1, 10, 100, 1000]

# for i in C:
#     logreg = LogisticRegression(C=i)
#     scores = cross_val_score(logreg, clean_df, target, cv=10)
#     print("Full features: When C = " + str(i) + ". Mean of the scores: {:.2f}".format(scores.mean()))
#     scores = cross_val_score(logreg, clean_df2, target, cv=10)
#     print("Reduced features: When C = " + str(i) + ". Mean of the scores: {:.2f}".format(scores.mean()))

# for i in range(1, 100):
#     logreg = LogisticRegression(C=i)
#     scores = cross_val_score(logreg, clean_df, target, cv=10)
#     print("Full features: When C = " + str(i) + ". Mean of the scores: {:.2f}".format(scores.mean()))
#     scores = cross_val_score(logreg, clean_df2, target, cv=10)
#     print("Reduced features: When C = " + str(i) + ". Mean of the scores: {:.2f}".format(scores.mean()))
    
# for i in range(1, 101):
#     logreg = LogisticRegression(C=i).fit(X_train, y_train)
#     print("When C is equal to " + str(i) + " training set result : " + str(logreg.score(X_train, y_train)))
#     print("When C is equal to " + str(i) + " test set result : " + str(logreg.score(X_test, y_test)))






# logreg = LogisticRegression(C=84)#.fit(X_train, y_train)
# # logreg.fit(X_test, y_test)

# scores = cross_val_score(logreg, X_test, y_test, cv=10)
# print("Full features: mean of the scores: {:.2f}".format(scores.mean()))

# scores = cross_val_score(logreg, clean_df, target, cv=10)
# print("Dataset with 36 features scores: {}".format(scores))
# print("Mean of the scores: {:.2f}".format(scores.mean()))

# scores = cross_val_score(logreg, clean_df2, target, cv=10)
# print("Dataset with 20 features scores: {}".format(scores))
# print("Mean of the scores: {:.2f}".format(scores.mean()))



# If we're overfitting 
# When the test and training set score are close it means I am likely underfitting
# print("Training set score " + str(logreg.score(X_train, y_train)))
#print("Test set score " + str(logreg.score(X_test, y_test)))
# You will need to explain alpha and regularization in this section, not the lit review

# C is changed, this relates to regularization I think, talk about this, this means
# less regularization
# logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
# print("Training set score: {:.3f}".format(logreg100.score(X_train, y_train)))
# print("Test set score: {:.3f}".format(logreg100.score(X_test, y_test)))

# C is set to 0.01, this means even more regularization
# logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
# print("Training set score: {:.3f}".format(logreg001.score(X_train, y_train)))
# print("Test set score: {:.3f}".format(logreg001.score(X_test, y_test)))

