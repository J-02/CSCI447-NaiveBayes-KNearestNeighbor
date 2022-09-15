import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Naivetest as Nt
import DataScrambler as ds
import os
from IPython.display import display
from tqdm import tqdm

# Creates training data using dataframe sample
# Creates testing data by dropping testing data from original data

def run(file):
    p = []
    r = []
    current = Nt.NaiveBayes('Data/' + file)
    fold1 = current.df.groupby('class', group_keys=False).apply(lambda x: x.sample(frac=.5))
    fold2 = current.df.drop(fold1.index)
    trainP = current.train(fold1)
    p.append(current.test(fold2, trainP))
    r.append(current.test(fold2, trainP))
    trainP = current.train(fold2)
    p.append(current.test(fold1, trainP))
    r.append(current.test(fold1, trainP))
    return np.average(r),np.average(p)

# Does same thing as run just with shuffling 10% of the features
def runS(file):
    p = []
    r = []
    current = Nt.NaiveBayes('Data/' + file)
    fold1 = current.df.groupby('class', group_keys=False).apply(lambda x: x.sample(frac=.5))
    fold2 = current.df.drop(fold1.index)
    trainP = current.train(ds.shuffle(fold1))
    p.append(current.test(fold2, trainP))
    r.append(current.test(fold2, trainP))
    trainP = current.train(ds.shuffle(fold2))
    p.append(current.test(fold1, trainP))
    r.append(current.test(fold1, trainP))
    return np.average(r), np.average(p)



# Does 5x2 CrossValidation for each data set

def CV(file):
    Pcontrol = []
    Rcontrol = []
    Pscrambled = []
    Rscrambled = []
    for i in range(2):  # first iteration non scrambled, second iteration scrambled.
        for x in range(5):
            if i != 1:
                Recall, Precision = run(file)
                name = file[0:-5]
                Pcontrol.append(Precision)
                Rcontrol.append(Recall)

            else:
                Recall, Precision = runS(file)
                Pscrambled.append(Precision)
                Rscrambled.append(Recall)

    scrambledP = pd.Series(Pscrambled, name='Scrambled Precision')
    controlP = pd.Series(Pcontrol, name='Control Precision')
    scrambledR = pd.Series(Rscrambled, name='Scrambled Recall')
    controlR = pd.Series(Rcontrol, name='Control Recall')
    data = [controlP, scrambledP, controlR, scrambledR]
    results = pd.DataFrame(data).transpose()
    print(name)
    print(results.to_string())
def allFiles():
    Pcontrol = []
    Rcontrol = []
    Pscrambled = []
    Rscrambled = []
    files = []
    a = False
    b = False
    for file in os.listdir("Data"):
        if file.endswith('.data'):
            CV(file)




allFiles()

def Video():
    pass