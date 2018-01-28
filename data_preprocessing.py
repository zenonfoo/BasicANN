### Data Extraction for BCC and non BCC ###
import h5py
import numpy as np
import pandas as pd

pd.set_option('display.width',320)

folder_name = './RamanData/tissue_'

# Finding the unique keys for all the tissue files -  to identify callable keys when pre-processing data
def findUniqueKeys(folder_name):

    unique_keys = []

    for x in range(3,72):

        # For the tissues with only a single task
        try:
            mat = h5py.File(folder_name + str(x) + '.mat')

            for item in list(mat.keys()):
                unique_keys.append(item)

        except OSError:
            print(x, 'OSError')

        except KeyError:
            print(x, 'KeyError')

        # For the tissues with multiple task
        try:

            for y in range(1,3):
                mat = h5py.File(folder_name + str(x) + '.mat')

                for item in list(mat.keys()):
                    unique_keys.append(item)

        except OSError:
            print(x, y, 'OSError')

        except KeyError:
            print(x, y, 'KeyError')

    return np.unique(unique_keys)

# Given an input key, this function will return the tissues which contain that key
def findTissue(folder_name,key):

    tissue = []

    for x in range(3,72):

        try:
            mat = h5py.File(folder_name + str(x) + '.mat')

            if key in list(mat.keys()):
                tissue.append()

        except OSError:
            print(x, 'OSError')

        except KeyError:
            print(x, 'KeyError')

    return tissue


# Returns 3 lists - Map of label indicated , Raman Data, Tissues Used
# Map of label - in the form of a list of 2D matrices
# Raman Data - in the form of a list of 3D matrices
def preProcess1(folder_name,testing_label):

    # Initializing variables
    label = []
    RamanData = []
    tissues_used = []

    for x in range(3, 72):

        try:
            # Opening file and obtaining keys
            mat = h5py.File(folder_name + str(x) + '.mat')
            keys = list(mat.keys())

            # Checking just for BCC data in all the files
            if testing_label in keys:
                label_map = np.array(mat[testing_label])
                RamanMap = np.array(mat['map_t' + str(x)])

            else:
                continue

            # Cheking for dimension mismatch
            # If there is no dimension mismatch organise data to be raman input and BCC/NonBCC cell output
            if label_map.shape[0] == RamanMap.shape[1] and label_map.shape[1] == RamanMap.shape[2]:
                label.append(label_map)
                RamanData.append(RamanMap)
                tissues_used.append(x)

        except OSError:
            print(x, 'OSError')

        except KeyError:
            print(x, 'KeyError')

        except IndexError:
            print(x, 'IndexError')

    return label, RamanData, tissues_used


# This is for the tissues with multiple tasks
# Returns 3 lists - Map of Label, Raman Data, Tissues Used
# Map of Label - in the form of a list of 2D matrices
# Raman Data - in the form of a list of 3D matrices
def preProcess2(folder_name,testing_label):

    # Initializing variables
    label = []
    RamanData = []
    tissues_used = []

    for x in range(3, 72):

        for y in range(1,3):

            try:
                # Opening file and obtaining keys
                mat = h5py.File(folder_name + str(x) + '_' + str(y) +'.mat')
                keys = list(mat.keys())

                # Checking just for BCC data in all the files
                if testing_label in keys:
                    label_map = np.array(mat[testing_label])
                    RamanMap = np.array(mat['map_r' + str(x) + '_' + str(y)])

                else:
                    continue

                # Cheking for dimension mismatch
                # If there is no dimension mismatch organise data to be raman input and BCC/NonBCC cell output
                if label_map.shape[0] == RamanMap.shape[1] and label_map.shape[1] == RamanMap.shape[2]:
                    label.append(label_map)
                    RamanData.append(RamanMap)
                    tissues_used.append(str(x) + '_' + str(y))

            except OSError:
                print(x, 'OSError')

            except KeyError:
                print(x, 'KeyError')

            except IndexError:
                print(x, 'IndexError')

    return label, RamanData, tissues_used


# Looks at the dimensions of each tissue data to initalize matrix to store them in one variable
def initalizeMatrix(label):
    row_size = 0

    for item in range(len(label)):
        row_size += label[item].shape[0] * label[item].shape[1]

    return row_size

# Converts all the tissue data into a singular 2D matrix variable to be ready for the neural network
def organiseData(label,raman):

    row_size = initalizeMatrix(label)
    data = np.zeros((row_size,1025))
    counter = 0

    for item in range(len(label)):
        for row in range(0, label[item].shape[0]):
            for column in range(0, label[item].shape[1]):

                data[counter,:-1] = raman[item][:, row, column]
                data[counter,-1] = label[item][row, column]
                counter += 1

    return data

def convert(matrix,index):

    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            if matrix[x,y] == 1:
                matrix[x,y] = index

    return matrix

def preProcessAllLabels(folder_name):

    # Initializing variables
    known_labels = ['bcc','dermis','fat','epi']
    label = []
    RamanData = []
    tissues_used = []

    for x in range(3, 72):

        try:
            # Opening file and obtaining keys
            mat = h5py.File(folder_name + str(x) + '.mat')
            keys = list(mat.keys())

            # Checking just for BCC data in all the files
            present_labels = [item for item in keys if item in known_labels]

            if present_labels:
                for key in present_labels:

                    label_map = np.zeros(mat[key].shape)

                    if key == 'bcc':

                        label_map = label_map + np.array(mat[key])

                    elif key == 'dermis':

                        temp = convert(np.array(mat[key]),2)
                        label_map = label_map + temp

                    elif key == 'fat':

                        temp = convert(np.array(mat[key]),3)
                        label_map = label_map + temp

                    elif key == 'epi':

                        temp = convert(np.array(mat[key]),4)
                        label_map = label_map + temp

                RamanMap = np.array(mat['map_t' + str(x)])

            else:
                continue

            # Cheking for dimension mismatch
            # If there is no dimension mismatch organise data to be raman input and BCC/NonBCC cell output
            if label_map.shape[0] == RamanMap.shape[1] and label_map.shape[1] == RamanMap.shape[2]:
                label.append(label_map)
                RamanData.append(RamanMap)
                tissues_used.append(x)

        except OSError:
            print(x, 'OSError')

        except KeyError:
            print(x, 'KeyError')

        except IndexError:
            print(x, 'IndexError')

    return label, RamanData, tissues_used


# Saving Data
np.save('BCC_Data_2',data)
np.save('Tissues_Used_2',tissue_used)




