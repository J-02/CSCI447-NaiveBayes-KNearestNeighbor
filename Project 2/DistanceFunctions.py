import pandas as pd
import numpy as np
import CrossValidation as cv
import time
from numba import njit

# Value difference metric
#---------------------------------------------------------------------
# takes two data points and finds the differences between each discrete feature returns aggregate distance between features
# if feature not on feature difference matrix defaults to 0 for distance
def depVDM(data, X, Y, p=2):
    FDMs = FDM(data) #
    d = 0
    for i in range(len(FDMs)):
        matrix = FDMs[i]
        matrix.columns = matrix.columns.astype(str)
        matrix.index = matrix.index.astype(str)

        x = str(X.iloc[0][i])



        y = str(Y.iloc[0][i])

        #print(matrix)
        if x in matrix.columns and y in matrix.columns:
            dist = matrix.loc[x,y]
        else:
            dist = 0
        d += np.power(dist,(1/p))
    return d

# FeatureDifferenceMatrix: input is dataframe
# given data constructs a matrix for each feature of differences between values
# returns list of dataframes corresponding to index of feature
# p is set to 2 by default and can be changed
# used to calculate VDM
def FDM(data,p=2):

    FDMs = []

    classes = data['class'].unique()
    columns = data.columns[data.columns != 'class']
    for col in columns:
        if data[col].dtype != "float64":
            matrix = pd.DataFrame()
            for Class in classes:
                pvalues = []
                for value in data[col].unique():
                    c = data[data[col] == value]['class'].count()
                    cc = data.loc[(data[col] == value) & (data['class'] == Class)].shape[0]
                    P = cc/c
                    pvalues.append(P)
                pvalues = np.power(np.absolute((np.subtract.outer(pvalues, pvalues))),p)
                if matrix.size == 0:
                    matrix = pd.DataFrame(data=pvalues, index=data[col].unique(), columns=data[col].unique())
                else:
                    matrix = np.add(pvalues,matrix)
            FDMs.append(matrix)
    return FDMs
#-----------------------------------------------------------

# new vdm:

def initialize(data):

    probabilities = {}
    for c in data['class'].unique():
        for feature in data.columns[data.columns != 'class']:
            values = data[feature].unique()
            for value in values:
                ID = str(feature) + ", " + str(value) + ', ' + str(c)
                probability = getProb(data, feature, value, c)
                probabilities[ID] = probability
    return probabilities
def VDM(data, x, y, probabilities = {}):
    x = x.to_dict('records')[0]
    y = y.to_dict('records')[0]
    totaldist = 0
    for k,v in x.items():
        if k == 'class':
            continue
        xprob = 0
        yprob = 0
        dist = 0
        for i in data['class'].unique():
            if probabilities:
                ID = str(k) + ", " + str(x[k]) + ', ' + str(i)
                if ID in probabilities.keys(): xprob = probabilities[ID]
                ID = str(k) + ", " + str(y[k]) + ', ' + str(i)
                if ID in probabilities.keys(): yprob = probabilities[ID]
            else:
                xprob = getProb(data, k, v, i)
                yprob = getProb(data, k, y[k], i)

            dist += abs(xprob-yprob)**2
        totaldist += dist**(1/2)
    return totaldist


def getProb(data,V_i, v_i, c_a): # gets P(Ca | Vi)
    x = data.loc[data[V_i] == v_i]  # Finds rows with the given feature value
    y = x.loc[data['class'] == c_a]  # Finds rows matching class that have the given feature and value
    z = data[data['class'] == c_a]  # Finds only rows containing the given class
    c_aCount = len(z)
    c_aANDv_i = len(y)
    prob = c_aANDv_i / c_aCount
    return prob


def EuclideanD(x,y):
    distance = 0
    for i in range(x.shape[1]-1):
        distance += abs(x.iloc[0,i]-y.iloc[0,i])
    distance = distance**(1/2)
    return distance


def kindaNN():
    data = cv.getSamples('breast-cancer-wisconsin.data')
    train = pd.DataFrame()

    for i in range(len(data)):

        if train.size == 0:
            train = data[i]
        else:
            train = train.append(data[i])

        samples = data[-1].sample(n=2)
        x = samples.iloc[[0]]
        y = samples.drop(x.index)

        print(x)
        print(y)
        start = time.perf_counter()
        print("v1 Distance:", depVDM(train, x, y))
        end = time.perf_counter()
        ms = (end - start) * 10 ** 6
        print(f"Elapsed {ms:.03f} micro secs.")
        start = time.perf_counter()
        print("v2 Distance:", VDM(train, x, y))
        end = time.perf_counter()
        ms = (end - start) * 10 ** 6
        print(f"Elapsed {ms:.03f} micro secs.")
        start = time.perf_counter()
        p= initialize(train)
        end = time.perf_counter()
        ms = (end - start) * 10 ** 6
        print("Initialization:",f"Elapsed {ms:.03f} micro secs.")

        start = time.perf_counter()
        print("v3 Distance:", VDM(train, x, y, p))
        end = time.perf_counter()
        ms = (end - start) * 10 ** 6
        print(f"Elapsed {ms:.03f} micro secs.")

#kindaNN()