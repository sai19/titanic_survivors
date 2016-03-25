# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 22:41:17 2015

@author: Saiprasad
"""

from __future__ import division
import re
import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier



# A function to get the title from a name.
def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""



titanic = pd.read_csv(r"E:\Practice\Titanic\Data\train.csv")
titles = titanic["Name"].apply(get_title)
title_mapping = {"Dona":1,"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles == k] = v
titanic["Title"] = titles

# The titanic variable is available here.
# filling all the missing values by it's median value
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median());
# Replace all the occurences of male with the number 0.
# and femALE WITH 1
titanic.loc[titanic["Sex"] == "male", "Sex"] = 1;
titanic.loc[titanic["Sex"] == "female", "Sex"] = 0;
## similarly change the boarding point too
titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0;
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1;
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2;
titanic["FamilySize"] = titanic["SibSp"]+(titanic["Parch"]);
# The .apply method generates a new series
titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x));
adult = (titanic["Age"]>16);
male = (titanic["Sex"]>0);
#titanic["ClassGender"] = adult*(male*(titanic["Fare"]/titanic["Pclass"]) + (titanic["Pclass"]==3)/(titanic["Fare"]+0.01)) + (titanic["Pclass"]==3)
titanic_female = titanic[titanic["Sex"]==1];
#titanic_female = titanic_female[titanic_female["Pclass"]==3];
predictors = ["Embarked","FamilySize","Title"];
# Initialize our algorithm class
#alg = GradientBoostingClassifier(random_state=5, n_estimators=100, max_depth=3);
alg = svm.SVC(kernel='rbf',gamma=0.0001,C=40);
kf = KFold(titanic_female.shape[0], n_folds=3, random_state=3)
predictions = []
for train, test in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (titanic_female[predictors].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = titanic_female["Survived"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(titanic_female[predictors].iloc[test,:])
    predictions.append(test_predictions)
predictions = np.concatenate(predictions, axis=0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0
scores = cross_validation.cross_val_score(alg, titanic_female[predictors], titanic_female["Survived"], cv=3)
print(scores.mean())

