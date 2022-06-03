import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
"""## Importing the dataset"""

dataset = pd.read_csv('C:/Users/nandi/final_project/test/Distance/ds.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

"""## Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

"""## Feature Scaling"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""## Training the Decision Tree Classification model on the Training set"""

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

"""## Predicting a new result"""
ddd=(classifier.predict(sc.transform([[21,	77,	48,	48,	48,0,0,	48,	48,	48,	48,	0]])))

"""# Saving model to disk"""
pickle.dump(classifier, open('Distance/model.pkl','wb'))
