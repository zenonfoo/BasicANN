import neural_network_testing as testing
import neural_network_training as training
import data_preprocessing as preprocess
import numpy as np

### Preprocessing Data ###
# Initializing Variables
folder = './RamanData/tissue_'
label = 'bcc'

# Loading Data
label_data,raman_data,tissues_used = preprocess.preProcessBCC(folder_name=folder,testing_label=label)

# Excluding one image for testing while the rest is used for
X_label = label_data[:-1]
X_raman = raman_data[:-1]
y_label = label_data[-1]
y_raman = raman_data[-1]

# Organising Data Into 2D Matrix
print('Organising Data Into 2D Matrix')
X,data_shape_X = preprocess.organiseData(X_label,X_raman)
y,data_shape_y = preprocess.organiseData(y_label,y_raman)

# Feature Scaling
print('Normalizing Data')
X_train,X_test,sc = training.normalize(X[:,:-1],y[:,:-1])

# Reverting back to list of 3D matrix form
print('Reverting')
new_X_raman = preprocess.revert(X_train,data_shape_X)
new_y_raman = preprocess.revert(X_test,data_shape_y)
new_y_raman = new_y_raman[0]

# Deleting variables to reduce RAM load
del label_data
del raman_data
del tissues_used
del X
del y
del data_shape_X
del data_shape_y
del X_train
del X_test
del X_raman
del y_raman

# Obtaining Grid Data
print('Obtaining Grid Data')
non_overlap_X = preprocess.obtainNonOverlapGridData(X_label,new_X_raman,3)
overlap_X = preprocess.obtainOverlapGridData(X_label,new_X_raman,3)
overlap_y = preprocess.obtainOverlapGridData(y_label,new_y_raman,3)

### Training ###
# Loading Data if data is already saved
print('Loading Data')
folder_name = 'BCC&NoBCC_Classification/2/Point_Test/BCC_Data_2.npy'
data,X,y = training.import_data(folder_name)

# Splitting dataset into training and test set
print('Splitting Data')
X_train, X_test, y_train, y_test = training.split(X,y)

# Feature Scaling
# sc variable is to be used later on to fit testing data
print('Normalizing Data')
X_train,X_test,sc = training.normalize(X_train,X_test)

# Initializing parameters for neural network
parameters = training.create_paramaters(input_dim=9216,units=500,layers=4,initializer='uniform',
                                        validation_split=0,activation='sigmoid',output_activation='sigmoid',
                                        optimizer='adam',batch=5000,epochs=30)

# Training Neural Network
classifier,history = training.neural_network(X[:,:-1],X[:,-1],parameters)

### Testing ###
# ROC Curve
# Initializing thresholds
thresholds = np.arange(0.1,1,0.02)

# Generating ROC data
print('Generating ROC')
roc = testing.ROC(classifier,X_test,y_test,thresholds)





