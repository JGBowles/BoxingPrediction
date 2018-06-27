#https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

from sklearn.neural_network import MLPClassifier
import itertools

#Random Grid Search
# n_estimators = [int(x) for x in np.linspace(start=10, stop=500, num=10)]
# max_features = ['auto', 'sqrt']
# max_depth = [int(x) for x in np.linspace(1, 50, num = 10)]
# min_samples_leaf = [int(x) for x in np.linspace(start=1, stop=250, num=10)]

# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_leaf' : min_samples_leaf
#               }

# forest = RandomForestClassifier()

# rf_random = RandomizedSearchCV(estimator = forest, param_distributions = random_grid, n_iter = 100, cv = 3, n_jobs=-1, verbose=3, random_state=42)
# rf_random.fit(X_train, y_train)
# print(rf_random.best_params_)

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


hidden_layer_sizes = ([x for x in itertools.product((10, 20, 30, 40, 50, 53, 73, 100), repeat=1)] + \
                       [x for x in itertools.product((10, 20, 30, 40, 50, 100), repeat=2)] )
                       #[((len(clean_df.columns))+1,)] )]
#print(hidden_layer_sizes)
alpha = [0.01, 0.1, 1, 10, 100]
activation = ['identity', 'logistic', 'tanh', 'relu']
solver = ['lbfgs', 'sgd', 'adam']
learning_rate = ['constant', 'invscaling', 'adaptive']

random_grid = {'hidden_layer_sizes': hidden_layer_sizes,
               'alpha': alpha,
              'activation' : activation,
              'solver': solver,
              'learning_rate': learning_rate}

prec_metric = make_scorer(precision_score)
k = StratifiedKFold(n_splits=10, shuffle=False)

mlp = MLPClassifier()

mlp_random = RandomizedSearchCV(estimator = mlp, param_distributions=random_grid, scoring=prec_metric,
                               n_iter=100, cv=k, n_jobs=-1, verbose=3)

mlp_random.fit(clean_df, target)
print("Best parameters found: " + str(mlp_random.best_params_))
print("Mean test score from Randomized Search CV: " + str(mlp_random.cv_results_['mean_test_score'].mean()))


mlp2 = MLPClassifier()
prec_metric2 = make_scorer(precision_score)
k2 = StratifiedKFold(n_splits=10, shuffle=False)
mlp_random2 = RandomizedSearchCV(estimator=mlp2, param_distributions=random_grid, scoring=prec_metric2,
                               n_iter=100, cv=k2, n_jobs=-1, verbose=3)

mlp_random2.fit(clean_df[colnames_selected], target)
print("Best parameters found for K20: " + str(mlp_random2.best_params_))
print("Mean test score from Randomized Search CV for K20: " + str(mlp_random2.cv_results_['mean_test_score'].mean()))










# mlp = MLPClassifier(hidden_layer_sizes=(13, 1), random_state=42)
# mlp.fit(X_train, y_train)

# #print("Accuracy " + str(mlp.score(X_test, y_test)))

# scores = cross_val_score(mlp, X_test, y_test, cv=10)
# print("Full features: mean of the scores: {:.2f}".format(scores.mean()))

# scores = cross_val_score(mlp, X_test, y_test, cv=10)
# print("Full features: mean of the scores: {:.2f}".format(scores.mean()))

#mlp.fit(X_train, y_train)

#Random Grid Search
# n_estimators = [int(x) for x in np.linspace(start=200, stop=500, num=10)]
# max_features = ['auto', 'sqrt']
# max_depth = [int(x) for x in np.linspace(5, 100, num = 5)]
# max_depth.append(None)

# hidden_layer_sizes = [(7, 7), (128,), (128, 7), (5, 2)]

# param_grid = {
#     'hidden_layer_sizes': hidden_layer_sizes
# }

# cv_search = GridSearchCV(estimator = mlp, param_grid = param_grid, n_jobs=-1, verbose = 3)
# cv_search.fit(X_train, y_train)
# print(cv_search.best_params_)



# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth
#               }

# forest = RandomForestClassifier()

# rf_random = RandomizedSearchCV(estimator = forest, param_distributions = random_grid, n_iter = 100, cv = 3, n_jobs=-1, verbose=3, random_state=42)
# rf_random.fit(X_train, y_train)
# print(rf_random.best_params_)


# scores = cross_val_score(mlp, clean_df, target, cv=10)
# print("Full features: mean of the scores: {:.2f}".format(scores.mean()))

# print("Training Accuracy " + str(mlp.score(X_train, y_train)))
# print("Testing Accuracy " + str(mlp.score(X_test, y_test)))

# A common way to adjust parameters in a neural network is to first create a network
# that is large enough to overfit, making sure that the task can actually be learned by
# the network. Then, once you know the training data can be learned, either shrink the
# network or increase alpha to add regularization, which will improve generalization
# performance.

# Algorithms part - http://scikit-learn.org/stable/modules/neural_networks_supervised.html

# # Poor accuracy could be down to poor scaling, scale with minmax scaler and see
# # if there';s an improvement in accuracy 
# # Either use minmax scaler or scale from cristi vlad video, standardscaler
# # neural networks 3 
# # Decent accuracy 
# # By default the MLP uses 100 hidden nodes
#print("Accuracy " + str(mlp.score(X_test, y_test)))

# Reduced the number of hidden nodes - 10 hidden units
# mlp = MLPClassifier(hidden_layer_sizes=[10], random_state=42)
# mlp.fit(X_train, y_train)
# print("Accuracy " + str(mlp.score(X_test, y_test)))

# Two hidden layers now with 10 nodes each
# mlp = MLPClassifier(hidden_layer_sizes=[10, 10], random_state=42)
# mlp.fit(X_train, y_train)
# print("Accuracy " + str(mlp.score(X_test, y_test)))

# Experiment with the alpha some more 
# mlp = MLPClassifier(hidden_layer_sizes=[10, 10], alpha=1, random_state=42)
# mlp.fit(X_train, y_train)
# print("Accuracy " + str(mlp.score(X_test, y_test)))

# Also use AUC from April chen's video
# Assess with cross validation
# Test on both the 20 features aswell as all features
