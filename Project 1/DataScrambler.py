import pandas as pd
import numpy as np
import random as rndm
import os

# Used to introduce noise into each of the data files and output a new file
# Counts total features per file then divides features by 10 and rounds to the
# nearest whole number to ensure at least 10% of features are shuffled.
# Then selects a random column and shuffles all values in that column.
# This occurs for at least 10% of the columns/features.


def scramble():

    files = []
    for file in os.listdir("Data"):
        if file.endswith('.data') and not file.endswith('scrambled.data'):

            df = pd.read_csv('Data/'+file, index_col=0)
            df.name = file.split(sep='.')[0] +"-scrambled.data"
            features = df.shape[1] - 1 # counts total features excluding class column
            Rfeatures = -(-features // 10) # divides features by 10 and rounds up to the nearest whole number

            for r in range(Rfeatures):
                i = rndm.randint(0,features) # randomly picks column/feature index
                np.random.shuffle(df[df.columns[i]].values) # shuffles column/feature values
            df.to_csv("Data/"+df.name)


