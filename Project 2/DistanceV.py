import numpy as np
import warnings

import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)


# this is used to calculate distance from discrete values by finding the difference between their cond probs
# i would be nice to have another vdm that doesnt need to be initialized for ekk as it increases the runtime a bit


# initialize
# -------------------------
# input is the training data
# computes the conditional probability for each attribute value of the training data for each class
# returns two lists: one with a dataframe for each class and one with training data probabilities for each class
def initialize(data):

    p = [data.groupby('class')[feature].value_counts() / data.groupby('class')[feature].count() for feature in data.columns][:-1]
      # list makes dataframe of all class conditional probabilites, index is feature values

    t = [] # empty list for each class
    for z in data['class'].unique(): # goes through classes
        temp = data
        for x in range(len(temp.columns[temp.columns != 'class'])):  # goes through each feature
            temp = temp.replace({temp.columns[temp.columns != 'class'][x]: p[x][z].to_dict()}) # puts all training data to probabilities
        t.append(temp)
    return t, p  # returns training data as probs and a map to convert the test point to probabilities

# checks if conditional prob exists returns 0 if not
# my sexy solution to when a probability doesn't exist
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
    x = X.copy()  # creates copy of test vector
    xClass = x.pop('class')  # removes class and stores it might not be needed
    pa = np.array([t[i].to_numpy()[:,:-1] for i in range(len(t))]) # puts probabilities of dataset in array
    totaldist = np.empty_like(pa)
    l = []
    xlist = list(x.values())  # dict to list to lookup values in multi dim array
    xp = np.array([np.tile([lookupCatch(p,i,c,xlist[i]) for i in range(len(xlist))],(pa[0].shape[0], 1)) for c in t[0]['class'].unique()])

    # peforming euclidean distance between probs dunno why it starts with f
    fdiff = np.subtract(xp,pa)**2
    fsum= np.sum(fdiff, axis=0)
    distance = np.sum(fsum, axis=1)**(1/2)

    # means if different ahah
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
# not used as its an ez one liner but here it is
def Euclidean(t, x):
    return np.sum((x - t) ** 2, axis=1) ** (1 / 2)







