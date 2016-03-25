# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 22:14:08 2015

@author: Saiprasad
"""

#algorithms = [
#    [GradientBoostingClassifier(random_state=1, n_estimators=60, max_depth=3), predictors],
#    [LogisticRegression(random_state=1), predictors]
#]
#
## Initialize the cross validation folds
#kf = KFold(titanic.shape[0], n_folds=3, random_state=1)
#
#predictions = []
#for train, test in kf:
#    train_target = titanic["Survived"].iloc[train]
#    full_test_predictions = []
#    # Make predictions for each algorithm on each fold
#    for alg, predictors in algorithms:
#        # Fit the algorithm on the training data.
#        alg.fit(titanic[predictors].iloc[train,:], train_target)
#        # Select and predict on the test fold.  
#        # The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
#        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
#        full_test_predictions.append(test_predictions)
#    # Use a simple ensembling scheme -- just average the predictions to get the final classification.
#    test_predictions = (3*full_test_predictions[0] + full_test_predictions[1]) / 4
#    # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
#    test_predictions[test_predictions <= .5] = 0
#    test_predictions[test_predictions > .5] = 1
#    predictions.append(test_predictions)
#
## Put all the predictions together into one array.
#predictions = np.concatenate(predictions, axis=0)
## Compute accuracy by comparing to the training data.
#actual = titanic["Survived"];
#error = np.subtract(actual,predictions);
#accuracy = (len(error)-np.count_nonzero(error))/len(error)
#print(accuracy);
#print(accuracy)
# Take the mean of the scores (because we have one for each fold)
#alg.fit(titanic[predictors], titanic["Survived"])