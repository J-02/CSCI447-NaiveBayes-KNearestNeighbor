import pandas as pd
import numpy as np
import CrossValidation as cv


# Nearest Neighbor
# -----------------------
# Will add all nearest neighbor algorithms here
class NearestNeighbor:

    def __init__(self, data):
        self.samples = cv.getSamples(data)
        self.k = 5


# Classifies points in data set until the performance stops improving
    def EKNN(self,k,n):
        train = pd.DataFrame
        for i in len(self.samples)-1: train.append(self.samples[i])
        train.reset_index()
        for index, row in train.iterrows():
            if

    def correctC(self, estimate, actual, classification = True, eps):

        if classification:
            if estimate = a_x:
                return true
            else:
                return false
        if not classification:
            if estimate = actual +







