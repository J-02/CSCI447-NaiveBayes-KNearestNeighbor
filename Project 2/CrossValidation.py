import numpy as np
import pandas as pd
import os

# getSamples
# --------------------------------------
# Splits dataset into 10 stratified samples
# input is a .data file in the /Data/ folder
# returns 10 stratified dataframes
def getSamples(dataset):
    df = pd.read_csv("Data/"+dataset,index_col=0, header=0)

    samples = []

# Creates samples for classification

    if 'class' in df.columns:
        sample = df.groupby('class', group_keys=False).apply(lambda x: x.sample(frac=.1, replace=False))
        df = df.drop(sample.index)
        tune = sample
        for i in range(10):
            n = 1/(10-i)
            sample = df.groupby('class', group_keys=False).apply(lambda x: x.sample(frac=n, replace=False))
            df = df.drop(sample.index)
            samples.append(sample)

        return samples, tune

# Creates samples for regression

    else:
        df = df.sort_values(by=df.columns[-1])
        sample = df.iloc[0::10, :]
        df2 = df.drop(sample.index)
        tune = sample
        for i in range(10):
            sample = df2.iloc[i::10, :]
            samples.append(sample)

        return samples, tune

def CrossValidation():
    pass

