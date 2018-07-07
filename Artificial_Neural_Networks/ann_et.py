"""
Created on Wed Jul  4 09:42:18 2018
@author: edward
"""

# Artificial Neural Network

# Installation steps, see Installations_Instructions.pdf
# For GPU instructions see http://www.python36.com/how-to-install-tensorflow-gpu-with-cuda-9-2-for-python-on-ubuntu/


# Part 1 - Data Preprocessing #####################################################

# Importing the libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now lets make the ANN! #################################################

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer with Dropout
# Update to Keras 2 API: classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(activation = "relu", input_dim = 11, units = 6, kernel_initializer = "uniform"))
classifier.add(Dropout(p = 0.1))

# Adding second hidden layer
classifier.add(Dense(activation = "relu", units = 6, kernel_initializer = "uniform"))
classifier.add(Dropout(p = 0.1))

# Adding the output layer
classifier.add(Dense(activation = "sigmoid", units = 1, kernel_initializer = "uniform"))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, epochs = 100)

# Part 3 - Making the predictions and evaluating the model ########################

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Predicting ta single new observation
"""
Predict if the customer with the following information will leave the bank:
    Geography: France
    Credit Score: 600
    Gender: Male
    Age: 40
    Tenure: 3
    Balance: 60000
    Number of Products: 2
    Has Credit Card: Yes
    Is Active Member: Yes
    Estimated Salary: 50000
"""
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Part 4 - Evaluating, Improving, and Tuning the ANN

# Evaluating the ANN

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(activation = "relu", input_dim = 11, units = 6, kernel_initializer = "uniform"))
    classifier.add(Dense(activation = "relu", units = 6, kernel_initializer = "uniform"))
    classifier.add(Dense(activation = "sigmoid", units = 1, kernel_initializer = "uniform"))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size=10, epochs = 100)
from keras import backend as K
with K.tf.device('/gpu:0'):
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 3)
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
# Dropout regularization to reduce overfitting if needed



# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(activation = "relu", input_dim = 11, units = 6, kernel_initializer = "uniform"))
    classifier.add(Dense(activation = "relu", units = 6, kernel_initializer = "uniform"))
    classifier.add(Dense(activation = "sigmoid", units = 1, kernel_initializer = "uniform"))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
from keras import backend as K
with K.tf.device('/gpu:0'):
    grid_search= GridSearchCV(estimator=classifier,
                               param_grid=parameters,
                               scoring='accuracy',
                               cv = 10)
    grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

