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

        try:
            mat = h5py.File(folder_name + str(x) + '.mat')

            for item in list(mat.keys()):
                unique_keys.append(item)

        except OSError:
            print(x, 'OSError')

        except KeyError:
            print(x, 'KeyError')

    return np.unique(unique_keys)

# Returns 3 lists - BCC Label, Raman Data, Tissues Used
# BCC Label - in the form of a list of 2D matrices
# Raman Data - in the form of a list of 3D matrices
def preProcess1(folder_name):

    # Initializing variables
    BCCData = []
    RamanData = []
    tissues_used = []

    # Keys in the dataset
    testing = ['bcc']
    map = 'map_t'

    for x in range(3, 72):

        try:
            # Opening file and obtaining keys
            mat = h5py.File(folder_name + str(x) + '.mat')
            keys = list(mat.keys())

            # Checking just for BCC data in all the files
            if testing[0] in keys:
                bcc = np.array(mat[testing[0]])
                RamanMap = np.array(mat[map + str(x)])

            else:
                continue

            # Cheking for dimension mismatch
            # If there is no dimension mismatch organise data to be raman input and BCC/NonBCC cell output
            if bcc.shape[0] == RamanMap.shape[1] and bcc.shape[1] == RamanMap.shape[2]:
                BCCData.append(bcc)
                RamanData.append(RamanMap)
                tissues_used.append(x)

        except OSError:
            print(x, 'OSError')

        except KeyError:
            print(x, 'KeyError')

        except IndexError:
            print(x, 'IndexError')

    return BCCData, RamanData, tissues_used


# This is for the tissues with multiple tasks
# Returns 3 lists - BCC Label, Raman Data, Tissues Used
# BCC Label - in the form of a list of 2D matrices
# Raman Data - in the form of a list of 3D matrices
def preProcess2(folder_name):
    # Initializing variables
    BCCData = []
    RamanData = []
    tissues_used = []

    # Keys in the dataset
    testing = ['bcc']
    map = 'map_t'

    for x in range(3, 72):

        for y in range(1,3):

            try:
                # Opening file and obtaining keys
                mat = h5py.File(folder_name + str(x) + '_' + str(y) +'.mat')
                keys = list(mat.keys())

                # Checking just for BCC data in all the files
                if testing[0] in keys:
                    bcc = np.array(mat[testing[0]])
                    RamanMap = np.array(mat[map + str(x) + '_' + str(y)])

                else:
                    continue

                # Cheking for dimension mismatch
                # If there is no dimension mismatch organise data to be raman input and BCC/NonBCC cell output
                if bcc.shape[0] == RamanMap.shape[1] and bcc.shape[1] == RamanMap.shape[2]:
                    BCCData.append(bcc)
                    RamanData.append(RamanMap)
                    tissues_used.append(str(x) + '_' + str(y))

            except OSError:
                print(x, 'OSError')

            except KeyError:
                print(x, 'KeyError')

            except IndexError:
                print(x, 'IndexError')

    return BCCData, RamanData, tissues_used


# Looks at the dimensions of each tissue data to initalize matrix to store
def initalizeMatrix(bcc):
    row_size = 0

    for item in range(len(bcc)):
        row_size += bcc[item].shape[0] * bcc[item].shape[1]

    return row_size

#
def organiseData(bcc,raman):

    row_size = initalizeMatrix(bcc)
    data = np.zeros((row_size,1025))
    counter = 0

    for item in range(len(bcc)):
        for row in range(0, bcc[item].shape[0]):
            for column in range(0, bcc[item].shape[1]):

                data[counter,:-1] = raman[item][:, row, column]
                data[counter,-1] = bcc[item][row, column]
                counter += 1

    return data


# Saving Data
np.save('BCC_Data_2',data)
np.save('Tissues_Used_2',tissue_used)


