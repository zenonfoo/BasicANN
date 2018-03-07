### Principal Component Analysis ###
from sklearn.decomposition import PCA
import neural_network_training as training
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Loading Data if data
print('Loading Data')
folder_name = 'BCC&NoBCC_Classification/2/BCC_Data_2.npy'
data,X,y = training.import_data(folder_name)

# Feature Scaling
print('Normalizing Data')
sc = StandardScaler()
X = sc.fit_transform(X)

# Performing PCA
pca = PCA()
pca.fit(X)

# Plotting Explained Variance Ratio
plt.bar(range(1,6),pca.explained_variance_ratio_[:6])
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