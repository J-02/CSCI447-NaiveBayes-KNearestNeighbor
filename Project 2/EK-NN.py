import pandas as pd
import numpy as np
import CrossValidation as cv


# Edited Nearest Neighbor
# -----------------------
# Classifies points in data set until the performance stops improving
def EK_NN(data, k, e):
    samples = cv.getSamples(data)


