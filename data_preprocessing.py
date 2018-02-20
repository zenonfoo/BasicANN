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

# This function converts the ones in the matrix to the index - to be one hot encoded later
def convert(matrix,index):

    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            if matrix[x,y] == 1:
                matrix[x,y] = index

    return matrix

# Returns 3 lists - Map of labels indicated , Raman Data, Tissues Used
# Map of label - in the form of a list of 2D matrices
# Raman Data - in the form of a list of 3D matrices
def preProcessAllLabels(folder_name):

    # Initializing variables
    known_labels = ['bcc','dermis','fat','epi']
    label = []
    RamanData = []
    tissues_used = []

    for x in range(3, 72):

        for y in range(0,3):

            if y == 0:
                folder_number = str(x)

            else:
                folder_number = str(x) + '_' + str(y)

        try:
            # Opening file and obtaining keys
            mat = h5py.File(folder_name + folder_number + '.mat')
            keys = list(mat.keys())

            # Checking just for BCC data in all the files
            present_labels = [item for item in keys if item in known_labels]

            if present_labels:

                label_map = np.zeros(mat[present_labels[0]].shape)

                for key in present_labels:

                    if key == 'bcc':

                        temp = convert(np.array(mat[key]), 1)
                        label_map = label_map + temp

                    elif key == 'dermis':

                        temp = convert(np.array(mat[key]),2)
                        label_map = label_map + temp

                    elif key == 'fat':

                        temp = convert(np.array(mat[key]),3)
                        label_map = label_map + temp

                    elif key == 'epi':

                        temp = convert(np.array(mat[key]),4)
                        label_map = label_map + temp

                RamanMap = np.array(mat['map_t' + folder_number])

            else:
                continue

            # Cheking for dimension mismatch
            # If there is no dimension mismatch organise data to be raman input and BCC/NonBCC cell output
            if label_map.shape[0] == RamanMap.shape[1] and label_map.shape[1] == RamanMap.shape[2]:
                label.append(label_map)
                RamanData.append(RamanMap)
                tissues_used.append(folder_number)

        except OSError:
            print(x, OSError)

        # This is due to the tissue not having any of the keys
        except KeyError:
            print(x, KeyError)

        except IndexError:
            print(x, IndexError)

        # This error is due to different keys having different shapes
        except ValueError:
            print(x, ValueError)

    return label, RamanData, tissues_used


# This is for the tissues with multiple tasks
# Returns 3 lists - Map of Label, Raman Data, Tissues Used
# Map of Label - in the form of a list of 2D matrices
# Raman Data - in the form of a list of 3D matrices
def preProcessBCC(folder_name,testing_label):

    # Initializing variables
    label = []
    RamanData = []
    tissues_used = []

    for x in range(3, 72):

        for y in range(0,3):

            if y == 0:
                folder_number = str(x)

            else:
                folder_number = str(x) + '_' + str(y)

            try:
                # Opening file and obtaining keys
                mat = h5py.File(folder_name + folder_number +'.mat')
                keys = list(mat.keys())

                # Checking just for BCC data in all the files
                if testing_label in keys:
                    label_map = np.array(mat[testing_label])
                    RamanMap = np.array(mat['map_t' + folder_number])

                else:
                    continue

                # Cheking for dimension mismatch
                # If there is no dimension mismatch organise data to be raman input and BCC/NonBCC cell output
                if label_map.shape[0] == RamanMap.shape[1] and label_map.shape[1] == RamanMap.shape[2]:
                    label.append(label_map)
                    RamanData.append(RamanMap)
                    tissues_used.append(folder_number)

            except OSError:
                print(x, 'OSError')

            except KeyError:
                print(x, 'KeyError')

            except IndexError:
                print(x, 'IndexError')

    return label, RamanData, tissues_used


# Looks at the dimensions of each tissue data to initialize matrix to store them in one variable
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
        for row in range(label[item].shape[0]):
            for column in range(label[item].shape[1]):

                data[counter,:-1] = raman[item][:, row, column]
                data[counter,-1] = label[item][row, column]
                counter += 1

    return data

# Function that returns the number of grids within the data, given the grid size of data that we want
def initializeGrid(label,gridlength):

    num_of_grids = 0

    for item in range(len(label)):
        num_of_grids += (label[item].shape[0]//gridlength) * (label[item].shape[1]//gridlength)

    return num_of_grids

# Cut off edges so that we only get complete grid sizes
def cut_off(label,raman,gridlength):

    cut_label = []
    cut_raman = []

    for item in label:
        cut_label.append(item[:(item.shape[0]//gridlength)*gridlength,:(item.shape[1]//gridlength)*gridlength])

    for item in raman:
        cut_raman.append(item[:,:(item.shape[1]//gridlength)*gridlength,:(item.shape[2]//gridlength)*gridlength])

    return cut_label,cut_raman

# Convert the image data into a form where each row contains raman sepctra with one grid, where the
# centre grid is the label to be classified
def convert_grid(data,gridlength,orginial_data,dimensionalized_data,raman_length,row,column_index):

    current_row = row

    # For each raman data we have that have been stretched out into 2D
    for column in range(dimensionalized_data.shape[1]):

        data[row, column_index:column_index + raman_length] = dimensionalized_data[:, column]

        # If you've finished one row of grid shaped data
        if (column + 1) % (gridlength * orginial_data.shape[2]) == 0:
            row += 1
            column_index = 0
            current_row = row

        # If I reach the column length of the image
        elif (column + 1) % orginial_data.shape[2] == 0:
            row = current_row
            column_index += raman_length

        # If I reach the grid length
        elif (column + 1) % gridlength == 0:
            row += 1
            column_index = column_index - ((gridlength-1)*raman_length)

        else:
            column_index += raman_length

    return data,row

def obtainLabel(data,label,gridlength):

    # Because the grid will always be square the centres for the row and column will be the same
    first_centre_index = gridlength//2
    counter = 0

    for item in label:
        horz_centre = np.arange(first_centre_index,item.shape[0],gridlength)
        vert_centre = np.arange(first_centre_index,item.shape[1],gridlength)

        for row in horz_centre:
            for column in vert_centre:
                data[counter,-1] = item[row,column]
                counter += 1

    return data


# Main function for obtaining data in the grid format
def obtainGridData(label, raman, gridlength):
    row_size = initializeGrid(label, gridlength)

    # Initializing memory to store our input data for out network
    data = np.zeros((row_size, 1024 * (gridlength ** 2) + 1))

    # Cut off excess image
    label, raman = cut_off(label, raman, gridlength)

    # Initializing variable
    varying_row = 0

    # For each image that we have
    for item in raman:

        # Reduce by one dimension
        temp = item.reshape(1024,item.shape[1]*item.shape[2])

        # Initialize variable
        raman_length = 1024
        row = varying_row
        column_index = 0

        # Converting data to a form where it can be used as an input for the neural network
        data,varying_row = convert_grid(data,gridlength,item,temp,raman_length,row,column_index)

        # Providing the label of the centre grid for

    data = obtainLabel(data,label,gridlength)

    return data

# Saving Data
# np.save('BCC_Data_4',data)
# np.save('Tissues_Used_4',tissues_used)
# np.save('Keys_Used_4',keys_used)
