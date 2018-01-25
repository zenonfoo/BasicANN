## Neural Network fo classifying BCC and non BCC cells

import numpy as np

folder_name = 'BCC&NoBCC_Classification/BCC_Data_2.npy'

# Importing dataset
def import_data(folder_name):

    data = np.load(folder_name)
    X = data[:,:-1]
    y = data[:,-1]

    return data,X,y

# Seperating data - with the main goal of normalizing the dataset w.r.t only the non-bcc raman data
def seperate(data):

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

# Splitting dataset into training and test set & Feature Scaling
def split_normalize(X,y):

    # Splitting dataset into training and test set
    from sklearn.model_selection import train_test_split
    # random_state is property that if you don't set it everytime you run the function there is a different outcome
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)

    # Feature Scaling - using non-bcc Raman spectra as a base
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test, y_train, y_test

# Running ANN algorithm
def neural_network(X_train, X_test, y_train, y_test):

    # Making the ANN
    from keras.models import Sequential
    from keras.layers import Dense

    # Initializing the ANN
    classifier = Sequential()

    # Adding to the input layer and the first hidden layer
    # kernel_initializer refers to inital weights
    classifier.add(Dense(units=50, kernel_initializer='uniform', activation='sigmoid', input_dim=1024))

    # Adding second layer
    classifier.add(Dense(units=50, kernel_initializer='uniform', activation='sigmoid'))

    # Adding output layer
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    # Compiling the ANN
    # optimizer: algorithm you want to use to find the optimal set of weights, adam is stochastic gradient descent
    # loss: loss function - binary_crossentropy is the logarithmic loss function
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Fitting classifier to training set
    classifier.fit(X_train, y_train, batch_size=100, epochs=10)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred>0.5)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    return cm