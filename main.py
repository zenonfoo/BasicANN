import neural_network_testing as testing
import neural_network_training as training
import data_preprocessing as preprocess
import obtaining_data as obtain
import numpy as np

### Preprocessing Data ###
# Initializing Variables
folder = './RamanData/tissue_'
label = 'bcc'

# Loading Data
label_data,raman_data,tissues_used = obtain.preProcessBCC(folder_name=folder,testing_label=label)

### Training ###
# Loading Data if data is already saved
print('Loading Data')
folder_name = 'BCC&NoBCC_Classification/4/BCC_Data_4.npy'
data,X,y = training.import_data(folder_name)

# Splitting dataset into training and test set
print('Splitting Data')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,stratify=y)

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





