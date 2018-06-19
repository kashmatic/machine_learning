import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')
## Replace the missing date (?) with rediculously low number
df.replace('?', -99999, inplace=True)
## We do not need the id column, because it doesnt contibute anything
df.drop(['id'], 1, inplace=True)

## Define the x and y
x = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

## train and shuffle for 20% of test size
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

## Classifier is our model
classifier = neighbors.KNeighborsClassifier()
classifier.fit(x_train, y_train)
accuracy = classifier.score(x_test, y_test)

print(accuracy)

## Prediction, make sure that is not already present in the data
new_measure = np.array([[5,1,1,1,2,2,3,1,1]])
new_measure = new_measure.reshape(len(new_measure),-1)

## Predict the class for the new_measure
prediction = classifier.predict(new_measure)
print(prediction)
