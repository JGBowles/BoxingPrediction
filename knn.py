from sklearn.neighbors import KNeighborsClassifier

# for i in range(1, 11):
# knn = knn = KNeighborsClassifier(n_neighbors=9)
# #     knn.fit(X_train_selected, y_train)
# #     print("Normal:" + str(knn.score(X_test_selected, y_test)))

# scores = cross_val_score(knn, clean_df, target, cv=10)
# print(scores.mean())
    
# knn = knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train_selected, y_train)
# print("Actual reduced:" + str(knn.score(X_test_selected, y_test)))

#scores = cross_val_score(knn, X_test_selected, y_test, cv=10)
#print(scores.mean())

# knn = knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(clean_df2, target)
# print("Assumed reduced:" + str(knn.score(clean_df2, target)))


for i in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(knn, X_test, y_test, cv=10)
    print("Number of neighbors: " + str(i) + "\nDataset with 36 features scores: {}".format(scores))
    print("Mean of the RDS scores: {:.2f}".format(scores.mean()))
    # SelectKBest Results applied
    scores = cross_val_score(knn, X_test, target, cv=10)
    print("Number of neighbors: " + str(i) + "\nDataset with 20 features scores: {}".format(scores))
    print("Mean of the K20 scores: {:.2f}".format(scores.mean()))


# Benefit of using cross-validation:
# -	Train test split performs a random split, we could get lucky with the data split. 
# -	With cross validation each example will be in the training set exactly once. 
# -	We get a best case and a worst case scenario with the multiple folds as opposed to the one accuracy. 
# Another benefit of cross-validation as compared to using a single split of the data is
# that we use our data more effectively. When using train_test_split, we usually use
# 75% of the data for training and 25% of the data for evaluation. When using five-fold
# cross-validation, in each iteration we can use four-fifths of the data (80%) to fit the
# model. When using 10-fold cross-validation, we can use nine-tenths of the data
# (90%) to fit the model. More data will usually result in more accurate models.

# As the simple k-fold strategy fails here, scikit-learn does not use it for classification,
# but rather uses stratified k-fold cross-validation. In stratified cross-validation, we
# split the data such that the proportions between classes are the same in each fold as
# they are in the whole dataset, as illustrated in Figure 5-2:
# For example, if 90% of your samples belong to class A and 10% of your samples
# belong to class B, then stratified cross-validation ensures that in each fold, 90% of
# samples belong to class A and 10% of samples belong to class B.

# Talk about benefits of cross validation etc 
