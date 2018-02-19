import neural_network_training as training
import neural_network_testing as testing
# import data_preprocessing as preprocess
from keras import optimizers
import numpy as np
import pandas as pd

### Investigating different learning rate decays ###

### Preprocessing Data ###
# Initializing Variables
folder = './RamanData/tissue_'
label = 'bcc'

# Loading Data
label_data,raman_data,tissues_used = preprocess.preProcessBCC(folder_name=folder,testing_label=label)

# Processing Data
data = preprocess.organiseData(label=label_data,raman=raman_data)

### Training ###
# Separating input and output data
X = data[:, :-1]
y = data[:, -1]

# Loading Data if data is already saved
folder_name = 'BCC&NoBCC_Classification/2/BCC_Data_2.npy'
data,X,y = training.import_data(folder_name)

# Splitting dataset into training and test set
X_train, X_test, y_train, y_test = training.split(X,y)

# Feature Scaling
# sc variable is to be used later on to fit testing data
X_train,X_test,sc = training.normalize(X_train,X_test)

# Initializing different layers
# layers = np.arange(1,10)

# Initializing different units
# units = np.arange(100,1000,100)

# Initializing different learning rates
# rate = 0.01

# Initializing variable to store ROC Data
ROC_Data = []

# Initializing variable to store history data
history_data = []

# Trying different parameters
# for rate in learning_rate:

    # # sgd = optimizers.SGD(lr=rate)
    #
    # # Initializing parameters for neural network
    # parameters = training.create_paramaters(units=100, layers=3, initializer='uniform',
    #                                         validation_split=0, activation='sigmoid',
    #                                         optimizer='adam', epochs=10)
    #
    # # Training neural network
    # classifier, history = training.neural_network(X_train,y_train,parameters)
    #
    # history_data.append(history)
    #
    # ### Testing ###
    # # ROC Curve
    # # Initializing thresholds
    # thresholds = np.arange(0.1, 1, 0.02)
    #
    # # Generating ROC data
    # roc = testing.ROC(classifier, X_test, y_test, thresholds)
    #
    # # Generating ROC column
    # roc['Learning Rate'] = rate*len(thresholds)
    #
    # if len(ROC_Data) == 0:
    #     ROC_Data = roc
    # else:
    #     ROC_Data = pd.concat([ROC_Data,roc],axis=0)

