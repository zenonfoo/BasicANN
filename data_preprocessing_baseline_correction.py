## Polynomial Baseline Correction ##
import numpy as np

# Returns polynomial fitted datapoints
def polynomial(data):

    # Row length and column length of data
    row_length = data.shape[0]
    col_length = data.shape[1]

    # Initializing memory to store coefficients for each raman spectra
    coeff = np.zeros((row_length,3))

    # Initializing x datapoints
    x = np.arange(1024)

    # Initializing memory to store fitted datapoints
    datapoints = np.zeros((row_length,col_length))

    # Obtaining coefficients
    for item in range(row_length):

        coeff[item,:] = np.polyfit(x,data[item,:],2)

    # Calculating datapoints
    for item in range(col_length):

        first_coeff = coeff[:, 0] * (data[:, item] ** 2)
        second_coeff = coeff[:, 1] * data[:, item]
        third_coeff = coeff[:,2]

        datapoints[:,item] = first_coeff + second_coeff + third_coeff

    datapoints = data - datapoints

    return datapoints