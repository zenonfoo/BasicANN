### Data Extraction for BCC and non BCC ###
import h5py
import numpy as np
import matplotlib.pyplot as plt

items = dict()
testing = ['bcc', 'map_t']
tissues_used = []

# Looking through all the files that are compatible with h5py
# Ignoring the files with 2 tasks first
for x in range(3, 72):

    try:
        mat = h5py.File('./RamanData/tissue_' + str(x) + '.mat')

        # Checking just for BCC data in all the files
        if testing[0] in list(mat.keys()):
            bcc = np.array(mat[testing[0]])
            RamanMap = np.array(mat[testing[1] + str(x)])

            # Cheking for dimension mismatch
            # If there is no dimension mismatch organise data to be raman input and BCC/NonBCC cell output
            if bcc.shape[0] == RamanMap.shape[1] and bcc.shape[1] == RamanMap.shape[2]:

                # Keeping track of which tissue is used
                tissues_used.append('tissue_' + str(x))

                # Organising Data
                for row in range(0,bcc.shape[0]):
                    for column in range(0,bcc.shape[1]):
                        RamanData = RamanMap[:,row,column]
                        BCCData = bcc[row,column]

                        if row == 0 and column == 0:
                            data = np.append(RamanData,BCCData)

                        else:
                            temp_data = np.append(RamanData,BCCData)
                            data = np.concatenate([data,temp_data],axis=0)

            else:
                print(x, 'Dimension Mismatch')

    except OSError:
        print(x, 'OSError')

    except KeyError:
        print(x, 'KeyError')

    except ValueError:
        print(x, 'ValueError')

