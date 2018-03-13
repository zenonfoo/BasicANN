## Training Neural Network fo classifying BCC and non BCC cells

import numpy as np
import data_preprocessing as preprocess

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

# Preprocesses data
def gridPreprocessing(label_data,raman_data):

    # Excluding one image for testing while the rest is used for
    X_label = label_data[:-1]
    X_raman = raman_data[:-1]
    y_label = label_data[-1]
    y_raman = raman_data[-1]

    # Organising Data Into 2D Matrix
    print('Organising Data Into 2D Matrix')
    X, data_shape_X = preprocess.organiseData(X_label, X_raman)
    y, data_shape_y = preprocess.organiseData(y_label, y_raman)

    # Feature Scaling
    print('Normalizing Data')
    X_train, X_test, sc = normalize(X[:, :-1], y[:, :-1])

    # Reverting back to list of 3D matrix form
    print('Reverting')
    training_image_data = preprocess.revert(X_train, data_shape_X)
    testing_image_data = preprocess.revert(X_test, data_shape_y)

    # Obtaining Grid Data
    print('Obtaining Grid Data')
    overlap_X = preprocess.obtainOverlapGridData(X_label, training_image_data, 3)
    overlap_y = preprocess.obtainOverlapGridData(y_label, testing_image_data, 3)

    return overlap_X,overlap_y

# Splitting dataset into training and test set
def split(X,y):

    from sklearn.model_selection import train_test_split
    # random_state is property that if you don't set it everytime you run the function there is a different outcome
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)

    return X_train, X_test, y_train, y_test

# Feature Scaling
def normalize(X_train,X_test):

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train,X_test,sc

# Constructing eigenvalues and eigenvectors
def eigen(X_train):

    cov_mat = np.cov(X_train.T)
    eigen_vals, eigen_vectors = np.linalg.eig(cov_mat)

    return eigen_vals, eigen_vectors

# Constructing History class to store data on accuracy and losses as the network trains
import keras

class History(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))

# Creating parameters to be used as argument during calling of the training function
def create_paramaters(input_dim,units,layers,initializer,validation_split,activation,output_activation,optimizer,batch,epochs):

    parameters = {}
    parameters['input_dimension'] = input_dim
    parameters['units'] = units
    parameters['layers'] = layers
    parameters['initializer'] = initializer
    parameters['validation_split'] = validation_split
    parameters['activation'] = activation
    parameters['output_activation'] = output_activation
    parameters['optimizer'] = optimizer
    parameters['batch'] = batch
    parameters['epochs'] = epochs

    return parameters

# Running ANN algorithm
def neural_network(X_train, y_train, parameters):

    # Making the ANN
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout

    # Initializing the ANN
    classifier = Sequential()

    # Adding input and hidden layers
    for layer in range(parameters['layers']):

        if layer == 0:
            # kernel_initializer refers to inital weights
            classifier.add(Dense(units=parameters['units'], kernel_initializer=parameters['initializer'],
                                 activation=parameters['activation'], input_dim=parameters['input_dimension']))
            classifier.add(Dropout(rate=0.2))

        else:
            classifier.add(Dense(units=parameters['units'], kernel_initializer=parameters['initializer'],
                                 activation=parameters['activation']))
            classifier.add(Dropout(rate=0.2))

    # Adding output layer
    classifier.add(Dense(units=1, kernel_initializer=parameters['initializer'],
                         activation=parameters['activation']))

    # Initializing classes to store loss and accuracy history
    history = History()

    # Compiling the ANN
    # optimizer: algorithm you want to use to find the optimal set of weights, adam is stochastic gradient descent
    # loss: loss function - binary_crossentropy is the logarithmic loss function
    classifier.compile(optimizer=parameters['optimizer'], loss='binary_crossentropy', metrics=['accuracy'])

    # Fitting classifier to training set
    classifier.fit(X_train, y_train, batch_size=parameters['batch'], epochs=parameters['epochs'], callbacks=[history])

    return classifier,history