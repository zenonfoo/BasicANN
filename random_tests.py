### Principal Component Analysis ###
from sklearn.decomposition import PCA
import neural_network_training as training
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


# Loading Data if data
print('Loading Data')
folder_name = 'BCC&NoBCC_Classification/4/BCC_Data_4.npy'
data,X,y = training.import_data(folder_name)

# Feature Scaling
print('Normalizing Data')
sc = StandardScaler()
X = sc.fit_transform(X)

# Performing PCA
pca = PCA()
pca.fit(X)

# Plotting Explained Variance Ratio
plt.bar(range(1,10),pca.explained_variance_ratio_[:9],alpha=0.7)
plt.xlabel('Principal Component Index')
plt.ylabel('Explained Variance Ratio')
plt.legend()

# Plotting First Component
plt.plot(pca.components_[:,0],label='First Principal Component')
plt.plot(pca.components_[:,1],label='Second Principal Component')
plt.plot(pca.components_[:,2],label='Thrid Principal Component')
plt.grid()
plt.title('Principal Componenets When Applying PCA To The Raman Data')
plt.legend()

# Plotting components
for count in range(1,10):

    plt.subplot(3,3,count)
    plt.plot(pca.components_[count,:])
    plt.title('Principal Component ' + str(count))
    plt.grid()


# ### Plotting Random Things ###
# import matplotlib.pyplot as plt
# from random import randint
# # labels = ['None','BCC','Dermis','Fat','Epi','Dye']
# for item in range(9):
#
#     plt.subplot(3,3,item+1)
#
#         for num in range(100):
#
#             temp = randint(1,(label[item].shape[0]))
#
#             plt.plot(label[item][temp,:])
#
#         # plt.title(labels[item])
#         plt.grid()


### Baseline Correction - Polyfitting
import data_preprocessing_baseline_correction as baseCorrection
import neural_network_training as training

# Loading Data if data is already saved
print('Loading Data')
folder_name = 'BCC&NoBCC_Classification/4/BCC_Data_4.npy'
data,X,y = training.import_data(folder_name)

print('Obtaining Datapoints')
datapoints = baseCorrection.polynomial(X)

from pylab import bone,pcolor,colorbar,plot,show
for i in range(6):

    temp = np.zeros(40000)

    for j in range(40000):

        if wanted_data[j] == i:
            temp[j] = 1
        else:
            temp[j] = 0

    temp = temp.reshape(200,200)
    temp = np.rot90(temp)

    plt.subplot(2,3,i+1)
    plt.pcolor(temp)
    colorbar()
