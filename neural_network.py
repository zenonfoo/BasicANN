## Neural Network fo classifying BCC and non BCC cells

import pandas as pd
import numpy as np

# Importing dataset
data = np.load('BCC&NoBCC_Classification/BCC_Data.npy')
X = np.array([row[:-1] for row in data])
y = np.array([row[-1] for row in data])

# Splitting dataset into training and test set
from sklearn.model_selection import train_test_split
# random_state is property that if you don't set it everytime you run the function there is a different outcome
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Making the ANN
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# Adding to the input layer and the first hidden layer
# kernel_initializer refers to inital weights
classifier.add(Dense(units=100, kernel_initializer='uniform', activation='sigmoid', input_dim=1024))

# Adding second layer
classifier.add(Dense(units=100, kernel_initializer='uniform', activation='sigmoid'))

# Adding output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
# optimizer: algorithm you want to use to find the optimal set of weights, adam is stochastic gradient descent
# loss: loss function - binary_crossentropy is the logarithmic loss function
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting classifier to training set
classifier.fit(X_train, y_train, batch_size=100, epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)