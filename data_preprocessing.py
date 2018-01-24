### Data Extraction for BCC and non BCC ###
import h5py
import numpy as np
import time
import matplotlib.pyplot as plt

items = dict()
testing = ['bcc', 'map_t']
tissues_used = []

def preProcess(folder_name):

    start = time.time()
    BCCData = []
    RamanData = []
    tissues_used = []

    for x in range(3, 72):

        try:
            mat = h5py.File(folder_name + str(x) + '.mat')

            # Checking just for BCC data in all the files
            if testing[0] in list(mat.keys()):
                bcc = np.array(mat[testing[0]])
                RamanMap = np.array(mat[testing[1] + str(x)])

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

    elapsed = time.time() - start
    print(elapsed)

    return BCCData,RamanData,tissues_used

def initalizeArray(bcc):
    row_size = 0

    for item in range(len(bcc)):
        row_size += bcc[item].shape[0] * bcc[item].shape[1]

    return row_size

def organiseData(bcc,raman):

    row_size = initalizeArray(bcc)
    data = np.zeros((row_size,1025))
    counter = 0

    for item in range(len(bcc)):
        for row in range(0, bcc[item].shape[0]):
            for column in range(0, bcc[item].shape[1]):

                data[counter,:-1] = raman[item][:, row, column]
                data[counter,-1] = bcc[item][row, column]
                counter += 1

    return data


