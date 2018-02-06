## Neural Network fo classifying BCC and non BCC cells

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

folder_name = 'BCC&NoBCC_Classification/BCC_Data_2.npy'

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

# Running ANN algorithm
def neural_network(X_train, X_test, y_train, y_test):

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

    # Compiling the ANN
    # optimizer: algorithm you want to use to find the optimal set of weights, adam is stochastic gradient descent
    # loss: loss function - binary_crossentropy is the logarithmic loss function
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Fitting classifier to training set
    classifier.fit(X_train, y_train, batch_size=1000, epochs=10)

    return classifier


def assessSingleClassModel(classifier,X_test,threshold):

    prediction = classifier.predict(X_test)
    prediction = (prediction>threshold)

    return prediction

def assessMultiClassModel(classifier,X_test):

    # Predicting the Test set results
    prediction = classifier.predict(X_test)
    encoded_prediction = np.zeros(prediction.shape)

    for x in range(prediction.shape[0]):
        index = max(prediction[x,:])
        for y in range(prediction.shape[1]):

            if prediction[x,y] == index:
                encoded_prediction[x,y] = 1

    return encoded_prediction

def singleLabelConfusionMatrix(y_test,y_pred):

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    return cm

def multiLabelConfusionMatrix(y_test,y_pred):

     # Initializing memory for confusion matrix
     cm = np.zeros((y_test.shape[1],y_test.shape[1],))

     for x in range(y_test.shape[0]):

         row_test = list(y_test[x,:])
         row_pred = list(y_pred[x, :])

         test_index = row_test.index(max(row_test))
         pred_index = row_pred.index(max(row_pred))

         cm[test_index,pred_index] += 1

     return cm

def ROC(classifier,X_test,y_test):

    thresholds = np.arange(0.1,1,0.02)
    store = []

    for threshold in thresholds:
        prediction = assessSingleClassModel(classifier,X_test,threshold)
        cm = singleLabelConfusionMatrix(y_test,prediction)
        TPR = cm[1][1]/(cm[1][1]+cm[1][0])
        FPR = cm[0][1]/(cm[0][1]+cm[0][0])
        store.append((TPR,FPR,threshold))

    store = pd.DataFrame(store)
    store.columns = ['TPR','FPR','Threshold']

    return store