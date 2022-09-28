import pandas as pd
import numpy as np
import CrossValidation as cv

# Value difference metric
#---------------------------------------------------------------------
# takes two data points and finds the differences between each discrete feature returns aggregate distance between features
# if feature not on feature difference matrix defaults to 0 for distance
def VDM(data, X, Y, p=2):
    FDMs = FDM(data)
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



data = cv.getSamples('soybean-small.data')
train = pd.DataFrame()
for i in range(len(data)-1):
    if train.size == 0:
        train = data[i]
    else:
        train = pd.merge(train,data[i])

    samples = data[i+1].sample(n=2)
    x = samples.iloc[[0]]
    y = samples.drop(x.index)
    #print(x)
    #print(y)
    print(VDM(train,x,y))