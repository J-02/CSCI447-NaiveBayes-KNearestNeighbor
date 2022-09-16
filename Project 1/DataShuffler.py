import pandas as pd
import numpy as np
import random as rndm
import os

#----------------------------------------------------------------------------------
# shuffle(df)
# Introduces noise to the data
# Counts total features per dataset excluding class then divides the total features by 10 and rounds to the
# nearest whole number to ensure at least 10% of features are shuffled.
# Then selects a random column and shuffles all values in that column.
# This occurs until at least 10% of the columns/features are shuffled.


def shuffle(df):
    features = df.shape[1] - 1 # counts total features excluding class column
    Rfeatures = -(-features // 10) # divides features by 10 and rounds up to the nearest whole number
    for r in range(Rfeatures):
        i = rndm.randint(0,features) # randomly picks column/feature index
        np.random.shuffle(df[df.columns[i]].values) # shuffles column/feature values
    return df
