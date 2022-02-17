import numpy as np
import pandas as pd
import os
train = pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for col in train.columns:
    change = "transformed_"+col
    train[change] = le.fit_transform(train[col])
categorical_columns = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
       'ring-type', 'spore-print-color', 'population', 'habitat']
train = train.drop(categorical_columns,axis=1)
y = train['transformed_class']
X = train.drop(['transformed_class'],axis=1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,y_train)
y_pred = neigh.predict(X_test)
print("Accuracy of KNN with 3 neighbours is:",accuracy_score(y_test,y_pred))
