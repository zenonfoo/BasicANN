## Neural Network fo classifying BCC and non BCC cells

import numpy as np
import pandas as pd


folder_name = 'BCC&NoBCC_Classification/2/BCC_Data_2.npy'

# Importing dataset
def import_data(folder_name):

    data = np.load(folder_name)
    X = data[:,:-1]
    y = data[:,-1]

    return data,X,y

# Separating data - with the main goal of normalizing the dataset w.r.t only the non-bcc raman data
def separate(data):

    # Initializing matrix so store the bcc data and non bcc data
    row_data = data.shape[0]
    count_bcc = [1 for item in range(row_data) if data[item,-1] == 1]
    sum_bcc = sum(count_bcc)
    bcc_store = np.empty([sum_bcc,1025])
    non_bcc_store = np.empty([row_data-sum_bcc,1025])

    bcc_counter = 0
    non_bcc_counter = 0

    for item in range(row_data):

        if data[item,-1] == 1:
            bcc_store[bcc_counter,:] = data[item,:]
            bcc_counter += 1

        else:
            non_bcc_store[non_bcc_counter,:] = data[item,:]
            non_bcc_counter += 1

    return bcc_store[:,:-1],non_bcc_store[:,:-1]

# Counts the number of each label within the dataset
def count(labels):

    label_counter = {}

    for item in labels:
        if item not in label_counter:
            label_counter[item] = 1

        else:
            label_counter[item] += 1

    return label_counter

# Encode output
def encode(y):

    from keras.utils import np_utils
    return np_utils.to_categorical(y)

# Splitting dataset into training and test set
def split(X,y):

    from sklearn.model_selection import train_test_split
    # random_state is property that if you don't set it everytime you run the function there is a different outcome
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)

    return X_train, X_test, y_train, y_test

#Feature Scaling
def normalize(X_train,X_test):

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train,X_test

import keras

class History(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))

# Running ANN algorithm
def neural_network(X_train, y_train):

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

    # Adding third layer
    classifier.add(Dense(units=100, kernel_initializer='uniform', activation='sigmoid'))

    # Adding output layer
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    # Initializing classes to store loss and accuracy history
    history = History()

    # Compiling the ANN
    # optimizer: algorithm you want to use to find the optimal set of weights, adam is stochastic gradient descent
    # loss: loss function - binary_crossentropy is the logarithmic loss function
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Fitting classifier to training set
    classifier.fit(X_train, y_train, batch_size=1000, epochs=10, callbacks=[history])

    return classifier,history