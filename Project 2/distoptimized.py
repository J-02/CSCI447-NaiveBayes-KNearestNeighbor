import timeit

import pandas as pd
import numpy as np
from numpy.linalg import norm
import CrossValidation as cv
import time
from functools import wraps


def timeit(my_func):
    @wraps(my_func)
    def timed(*args, **kw):
        tstart = time.time()
        output = my_func(*args, **kw)
        tend = time.time()

        print('"{}" took {:.3f} ms to execute\n'.format(my_func.__name__, (tend - tstart) * 1000))
        return output

    return timed

# data is dataframe of train data not including test data
def initialize(data):
    p = [data.groupby('class')[feature].value_counts() / data.groupby('class')[feature].count() for feature in (data.columns[data.columns != 'class'])]
    return p

# data is same as initializeV2, x and y are both vectors, array, p is list of series
def VDMv2(data, x, y, p=[]):
    totaldist = 0
    for l in range(len(x)-1):
        t = x[l]
        u = y[l]
        xprob = 0
        yprob = 0
        dist = 0
        for i in data['class'].unique():
            o = p[l][i]
            if t in o.index:
                xprob = o[t]
            if u in o.index:
                yprob = o[u]
            dist += abs(xprob - yprob) ** 2
        totaldist += dist ** (1 / 2)
    return totaldist

def EuclideanVector(x,y):
    distance = norm(x-y)
    return distance


#data = pd.concat(cv.getSamples('breast-cancer-wisconsin.data')[0][0:9])


@timeit
def v2test():
    data = cv.getSamples('breast-cancer-wisconsin.data')[0]
    train = pd.concat(data[0:9])
    trainV = train.to_numpy()
    test = pd.DataFrame(data[9])
    testV = test.to_numpy()
    p = initializeV2(train)
    for x in testV:
        dist = [VDMv2(train,x,y,p) for y in trainV]
        train['Dist'] = dist
        neighbors = train.nsmallest(5, 'Dist')
    print(neighbors)

@timeit
def v1test():
    data = cv.getSamples('breast-cancer-wisconsin.data')[0]
    train = pd.concat(data[0:9])
    #trainV = train.to_numpy()
    test = pd.DataFrame(data[9])
    #testV = test.to_numpy()
    p = initialize(train)
    for index, row in test.iterrows():
        dist = [VDM(train,x,y,p) for y in train.iterrows()]
        train['Dist'] = dist
        neighbors = train.nsmallest(5, 'Dist')
    print(neighbors)



