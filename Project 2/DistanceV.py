import pandas as pd
import numpy as np
import time
import warnings
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
def initialize(data):

    p = [data.groupby('class')[feature].value_counts() / data.groupby('class')[feature].count() for feature in data.columns]
    l = pd.DataFrame(p).transpose()  # makes dataframe of all class conditional probabilites, index is feature values
    # and column is the feature
    ps = [l.loc[x,:].fillna(0) for x in data['class'].unique()]
    t = []
    for z in range(len(data['class'].unique())):
        temp = data
        for x in temp.columns[temp.columns != 'class']:
            temp = temp.replace({x: ps[z][x].to_dict()})
        t.append(temp)
    return t, ps


# VDM
# ------------------
# calculates distance between two vectors given the conditional probabilities of each attribute value for each class
# returns 1D array of distances the length of the training data
def VDM(t, x, p=[]):
    xClass = x['class']
    totaldist = 0
    xprob = 0
    yprob = 0
    pa = np.array([t[i].to_numpy()[:,:-1] for i in range(len(t))]) # puts probabilities of datset in array
    totaldist = np.empty_like(pa)

    xp = np.array([np.tile(np.array(p[i].lookup(list(x.values()),list(x))[:-1]),(pa[0].shape[0],1))  for i in range(len(t[0]['class'].unique()))]) # probabilities for the features of x for each class
    fdiff = np.subtract(xp,pa)**2
    fsum= np.sum(fdiff, axis=0)
    distance = np.sum(fsum, axis=1)**(1/2)

    return distance

# Euclidean
# ------------------
# returns the Euclidean distance between two vectors
# must have same amount of features
def Euclidean(t, x):
    return np.sum((x - t) ** 2, axis=1) ** (1 / 2)







