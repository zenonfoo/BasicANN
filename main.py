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

# Processing Data
data = preprocess.organiseData(label=label_data,raman=raman_data)

### Training ###
# Separating input and output data
X = data[:, :-1]
y = data[:, -1]

# Loading Data if data is already saved
print('Loading Data')
folder_name = 'BCC&NoBCC_Classification/2/BCC_Data_2.npy'
data,X,y = training.import_data(folder_name)

# Splitting dataset into training and test set
print('Splitting Data')
X_train, X_test, y_train, y_test = training.split(X,y)

# Feature Scaling
# sc variable is to be used later on to fit testing data
print('Normalizing Data')
X_train,X_test,sc = training.normalize(X_train,X_test)

# Obataining eigenvalues and eigenvectors
eigen_val, eigen_vector = training.eigen(X_train)

# Normalizing eigenvalues
tot = sum(eigen_val)
norm_eigen_val = [(i/tot) for i in sorted(eigen_val,reverse=True)]

# Plotting decreasing eigenvalues
import matplotlib.pyplot as plt
plt.bar(range(1,X_train.shape[1]+1),norm_eigen_val,alpha=0.5,align='center',label='individual explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')

# Initializing parameters for neural network
parameters = training.create_paramaters(input_dim=1024,units=500,layers=4,initializer='uniform',
                                        validation_split=0,activation='sigmoid',output_activation='sigmoid',
                                        optimizer='adam',batch=5000,epochs=30)

# Training Neural Network
classifier,history = training.neural_network(X_train,y_train,parameters)

### Testing ###
# Prediction
# prediction = testing.assessSingleClassModel(classifier,X_test)

# Confusion Matrix
# cm = testing.singleLabelConfusionMatrix(y_test,prediction)

# ROC Curve
# Initializing thresholds
thresholds = np.arange(0.1,1,0.02)

# Generating ROC data
print('Generating ROC')
roc = testing.ROC(classifier,X_test,y_test,thresholds)





