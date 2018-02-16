import neural_network_testing as testing
import neural_network_training as training
import data_preprocessing as preprocess
import numpy as np

### Preprocessing Data ###
# Initializing Variables
folder = './RamanData/tissue_'
label = 'bcc'

# Loading Data
label_data,raman_data,tissues_used = preprocess.preProcessBCC(folder_name=folder,testing_label=testing)

# Processing Data
data = preprocess.organiseData(label=label_data,raman=raman_data)

### Training ###
# Separating input and output data
X = data[:, :-1]
y = data[:, -1]

# Splitting dataset into training and test set
X_train, X_test, y_train, y_test = training.split(X,y)

# Feature Scaling
# sc variable is to be used later on to fit testing data
X_train,X_test,sc = training.normalize()

# Initializing parameters for neural network
parameters = training.create_paramaters(units=100,layers=3,initializer='uniform',
                                        validation_split=0.1,activation='sigmoid',
                                        optimizer='adam',epochs=10)

# Training Neural Network
classifier,history = training.neural_network(X_train,y_train)

### Testing ###
# Prediction
prediction = testing.assessSingleClassModel(classifier,X_test)

# Confusion Matrix
cm = testing.singleLabelConfusionMatrix(y_test,prediction)

# ROC Curve
# Initializing thresholds
thresholds = np.arange(0.1,1,0.02)

# Generating ROC data
roc = testing.ROC(classifier,X_test,y_test,thresholds)




