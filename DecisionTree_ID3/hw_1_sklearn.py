import sklearn.datasets as datasets
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# check for the sklearn version, it has to be 0.21
import sklearn
print(sklearn.__version__)
#Import the DecisionTreeClassifier
# from sklearn.tree import DecisionTreeClassifier
import pandas as pd


dataFrame = pd.read_csv('dt_data.txt', sep="[\s\d():,;\n]+", engine='python')
dataFrame.dropna(axis='columns', inplace=True)
dataFrame.rename(columns={'Beer':'Enjoy', 'Favorite': 'Beer'}, inplace=True)

"""
Split the data into a training and a testing set
"""

train_features = dataFrame.iloc[:15,:-1]
test_features = dataFrame.iloc[15:,:-1]
train_targets = dataFrame.iloc[:15,-1]
test_targets = dataFrame.iloc[15:,-1]

train_targets = pd.get_dummies(train_targets)
test_targets = pd.get_dummies(test_targets)

print(train_features)
print(train_targets)
print(test_features)
print(test_targets)
"""
Train the model
"""

tree = DecisionTreeClassifier(criterion='').fit(train_features,train_targets)

"""
Predict the classes of new, unseen data
"""
prediction = tree.predict(test_features)

"""
Check the accuracy
"""

print("The prediction accuracy is: ",tree.score(test_features,test_targets)*100,"%")