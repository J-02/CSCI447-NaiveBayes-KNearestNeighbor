import pandas as pd
import numpy as np
import CrossValidation as cv



# FeatureDifferenceMatrix: input is dataframe
# given data constructs matrix of differences between values
# has issues with building matricies properly, will try to use data frames
def FDM(data,p=2):

    FDMs = []

    classes = data['class'].unique()
    columns = data.columns[data.columns != 'class']
    for col in columns:
        if data[col].dtype != "float64":
            matrix = np.array([])
            for Class in classes:
                pvalues = []
                for value in data[col].unique():
                    c = data[data[col] == value]['class'].count()
                    cc = data.loc[(data[col] == value) & (data['class'] == Class)].shape[0]
                    P = cc/c
                    pvalues.append(P)
                pvalues = np.power(np.absolute((np.subtract.outer(pvalues, pvalues))),p)
                if matrix.size == 0:
                    matrix = pvalues
                else:
                    matrix = np.add(pvalues,matrix)
            FDMs.append(matrix)
    return FDMs

# Value difference metric
# takes two data points and finds the differences between each discrete feature
def VDM(data, X, Y, p=2):
    FDMs = FDM(data)
    d = 0
    for i in range(len(FDMs)):
        matrix = FDMs[i]
        x = X.iloc[0][i]-1
        y = Y.iloc[0][i]-1
        if matrix.shape[0] < x or matrix.shape[1] < y:
            dist = 10000
        else:
            dist = matrix[x][y]
        d += np.power(dist,(1/p))
    return d



data = cv.getSamples('soybean-small.data')
train = pd.DataFrame()
for i in range(data.__sizeof__()):
    if train.size == 0:
        train = data[i]
    else:
        train = pd.merge(train,data[i])

    x = data[i+1].sample(n=1)
    print(x)
    y = data[i+1].sample(n=1,random_state=1100)
    print(y)
    print(VDM(train,x,y))