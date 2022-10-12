import pandas as pd
import numpy as np
import time
import warnings
from line_profiler_decorator import profiler
warnings.simplefilter(action='ignore', category=FutureWarning)


def timeit(my_func):
    @wraps(my_func)
    def timed(*args, **kw):
        tstart = time.time()
        output = my_func(*args, **kw)
        tend = time.time()

        print('"{}" took {:.3f} ms to execute\n'.format(my_func.__name__, (tend - tstart) * 1000))
        return output

    return timed

# initialize
# -------------------------
# input is the training data
# computes the conditional probability for each attribute value of the training data for each class
# returns two lists: one with a dataframe for each class and one with training data probabilities for each class
#@profiler
def initialize(data):

    p = [data.groupby('class')[feature].value_counts() / data.groupby('class')[feature].count() for feature in data.columns][:-1]
      # list makes dataframe of all class conditional probabilites, index is feature values

    t = []
    for z in data['class'].unique():
        temp = data
        for x in range(len(temp.columns[temp.columns != 'class'])):
            temp = temp.replace({temp.columns[temp.columns != 'class'][x]: p[x][z].to_dict()}) # puts all training data to probabilities
        t.append(temp)
    return t, p

# checks if conditional prob exists returns 0 if not
def lookupCatch(p,i,c,v):
    try:
       return p[i][c][v]
    except:
        return 0

# VDM
# ------------------
# calculates distance between two vectors given the conditional probabilities of each attribute value for each class
# returns 1D array of distances the length of the training data
def VDM(t, X, p=[], means=None):
    x = X.copy()
    xClass = x.pop('class')
    totaldist = 0
    xprob = 0
    yprob = 0
    pa = np.array([t[i].to_numpy()[:,:-1] for i in range(len(t))]) # puts probabilities of datset in array
    totaldist = np.empty_like(pa)
    l = []
    xlist = list(x.values())
    xp = np.array([np.tile([lookupCatch(p,i,c,xlist[i]) for i in range(len(xlist))],(pa[0].shape[0], 1)) for c in t[0]['class'].unique()])

    fdiff = np.subtract(xp,pa)**2
    fsum= np.sum(fdiff, axis=0)
    distance = np.sum(fsum, axis=1)**(1/2)

    if means is None:
        return distance
    else:
        out = means.loc[t[0].index.values]
        out['dist'] = distance
        out = np.array(out.to_numpy())
        return out

# Euclidean
# ------------------
# returns the Euclidean distance between two vectors
# must have same amount of features
def Euclidean(t, x):
    return np.sum((x - t) ** 2, axis=1) ** (1 / 2)







