# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 10:18:30 2015

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
titanic["AgeSex"] = (titanic["Sex"])+ (1/((titanic["Age"])+0.01));
titanic["AgeFare"] = 1/((titanic["Fare"]+0.01)*(titanic["Age"]+0.01));
titanic["FareSex"] = (titanic["Sex"]) + (1/(titanic["Fare"]+0.01));
## similarly change the boarding point too
titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0;
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1;
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2;
titanic["FamilySize"] = titanic["SibSp"]+(titanic["Parch"]);
# The .apply method generates a new series
titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x));
titanic["nameSex"] = (1/titanic["NameLength"])+(titanic["Sex"]);
titanic["nameAge"] = (titanic["NameLength"])+0.5*(titanic["Age"]);
#titanic["AgeSexClass"] = 
#titanic["ClassGender"] = (titanic["Age"]>15)*((titanic["Sex"]>0)*( (titanic["Pclass"])/(titanic["Fare"]+0.01)*(titanic["Title"]+0.01)) + (titanic["Pclass"]==3)/(titanic["AgeFare"]) ) + (titanic["Pclass"]==3);
titanic["ClassGender"] = (titanic["Sex"]>0)*(1/(titanic["NameLength"]+0.01)) + ((titanic["Pclass"]>2)*(1/(titanic["FamilySize"]+0.01)));
# The columns we'll use to predict the target
#predictors = ["ClassGender","FareSex","nameSex","AgeFare","AgeSex","Title","Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked","FamilySize","NameLength"]
#predictors = ["ClassGender","FareSex","nameSex","AgeSex","Title","Sex","Fare"];
predictors = ["Pclass","Sex","Age","ClassGender"]
# Initialize our algorithm class
#alg = GradientBoostingClassifier(random_state=5, n_estimators=50, max_depth=10);
alg = svm.SVC(kernel='rbf',gamma=0.01,C=100);
#alg = RandomForestClassifier(n_estimators=10)
#alg = tree.DecisionTreeClassifier(min_samples_split=70);
#Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.
kf = KFold(titanic.shape[0], n_folds=3, random_state=3)
predictions = []
for train, test in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (titanic[predictors].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = titanic["Survived"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)
predictions = np.concatenate(predictions, axis=0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
print(scores.mean())

titanic_test = pd.read_csv(r"E:\Practice\Titanic\Data\test.csv")
titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median());
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic["Fare"].median());
# Replace all the occurences of male with the number 0.
# and femALE WITH 1
titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]
# The .apply method generates a new series
titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 1
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 0

## similarly change the boarding point too
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2


    
titles_test = titanic_test["Name"].apply(get_title)
for k,v in title_mapping.items():
    titles_test[titles_test == k] = v
titanic_test["Title"] = titles_test;

titanic_test["AgeSex"] = (titanic_test["Sex"]+0.01)+(1/(titanic_test["Age"])+0.01);
titanic_test["AgeFare"] = 1/((titanic_test["Fare"]+0.01)*(titanic_test["Age"]+0.01));
titanic_test["nameSex"] = (1/titanic_test["NameLength"])+(titanic_test["Sex"])
titanic_test["FareSex"] = (titanic_test["Sex"]) + (1/(titanic_test["Fare"]+0.01));
titanic_test["nameAge"] = (titanic_test["NameLength"])+0.5*(titanic_test["Age"]);
#titanic_test["ClassGender"] = (titanic_test["Age"]>15)*((titanic_test["Sex"]>0)*(titanic_test["Pclass"]/(titanic_test["Fare"]+0.01)*(titanic_test["Title"]+0.01))+(titanic_test["Pclass"]==3)/titanic_test["AgeFare"]) + (titanic_test["Pclass"]==3);
titanic_test["ClassGender"] = (titanic_test["Sex"]>0)*(1/(titanic_test["FamilySize"]+0.01)) + ((titanic_test["Pclass"]>2)*(1/(titanic_test["FamilySize"]+0.01)));
# Train the algorithm using all the training data
test_predictions = alg.predict(titanic_test[predictors])
# Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
#test_predictions[predictions <= .5] = 0
#test_predictions[predictions > .5] = 1
#print predictions;
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": test_predictions
    })
    
submission.to_csv(r"E:\Practice\Titanic\kaggle_Test.csv", index=False)
