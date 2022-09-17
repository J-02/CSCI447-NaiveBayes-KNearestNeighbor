import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Naivetest as Nt
import DataShuffler as ds
import os

# Hold-out experiments
# --------------------------------------------------------------------------------
# run:
#
# Creates random stratified training data using dataframe groupby and sample
# Creates testing data by dropping testing data from original data
# Once initial results are measured the training data and testing data are swapped and results are measured again
# The average of the hold-out experiment is output for the recall and precision loss functions


def run(file):
    p = []
    r = []
    # initialization and creating folds
    current = Nt.NaiveBayes('Data/' + file)
    fold1 = current.df.groupby('class', group_keys=False).apply(lambda x: x.sample(frac=.5))
    fold2 = current.df.drop(fold1.index)

    # performing first test
    trainP = current.train(fold1)
    results = current.test(fold2, trainP)
    p.append(results[1])
    r.append(results[0])

    # performing second test
    trainP = current.train(fold2)
    results = current.test(fold1, trainP)
    p.append(results[1])
    r.append(results[0])
    return np.average(r), np.average(p)

# runS:
#
# Does same thing as run just with shuffling 10% of the features before each test
# ds.shuffle is used from DataShuffler.py

def runS(file):
    p = []
    r = []
    current = Nt.NaiveBayes('Data/' + file)
    fold1 = current.df.groupby('class', group_keys=False).apply(lambda x: x.sample(frac=.5))
    fold2 = current.df.drop(fold1.index)
    trainP = current.train(ds.shuffle(fold1))
    results = current.test(fold2, trainP)
    p.append(results[1])
    r.append(results[0])
    trainP = current.train(ds.shuffle(fold2))
    results = current.test(fold1, trainP)
    p.append(results[1])
    r.append(results[0])
    return np.average(r), np.average(p)

# ------------------------------------------------------------------------------------------
# CV - Cross Validation
# Collects results from 5 experiments on the control dataset and 5 experiments on the shuffled dataset
# Does 5x2 CrossValidation for each data set


def CV(file):
    Pcontrol = []
    Rcontrol = []
    Pshuffled = []
    Rshuffled = []
    for i in range(2):  # first iteration non shuffled, second iteration shuffled.
        for x in range(5):
            if i != 1:
                Recall, Precision = run(file)
                name = file[0:-5]
                Pcontrol.append(Precision)
                Rcontrol.append(Recall)

            else:
                Recall, Precision = runS(file)
                Pshuffled.append(Precision)
                Rshuffled.append(Recall)

    data = [np.average(Pcontrol), np.average(Pshuffled), np.average(Rcontrol), np.average(Rshuffled)]
    return data

# ---------------------------------------------------------------------------------------------------
# allFiles
# Runs CV() on each dataset in the data folder and creates a dataframe and graph from the results
# Provides a summary output of our experiments


def allFiles():
    Pcontrol = []
    Rcontrol = []
    Pshuffled = []
    Rshuffled = []
    files = []
    for file in os.listdir("Data"):
        if file.endswith('.data'):
            data = CV(file)
            Pcontrol.append(data[0])
            Pshuffled.append(data[1])
            Rcontrol.append(data[2])
            Rshuffled.append(data[3])
            files.append(file[0:-5])

    shuffledP = pd.Series(Pshuffled, name='shuffled Precision')
    controlP = pd.Series(Pcontrol, name='Control Precision')
    shuffledR = pd.Series(Rshuffled, name='shuffled Recall')
    controlR = pd.Series(Rcontrol, name='Control Recall')
    Datasets = pd.Series(files, name='Dataset')
    data = [controlP, shuffledP, controlR, shuffledR]
    results = pd.DataFrame(data=np.transpose(data),
                           columns=['Control Precision', 'Shuffled Precision', 'Control Recall', 'Shuffled Recall'],
                           index=Datasets)
    #print(results.to_latex())
    #results.to_csv('Results/Results.data')
    data = [controlP, controlR, shuffledP, shuffledR]
    results = pd.DataFrame(data=np.transpose(data), columns=['Control Precision', 'Control Recall', 'Shuffled Precision', 'Shuffled Recall'], index=Datasets)

    # Bar chart plotting
    results.plot.bar()
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.xticks(rotation=0)
    plt.autoscale()
    #plt.savefig('Results/Results.png')
    plt.show()

# ----------------------------------------------------------------------------------------------------
# Video
# Running this code gives the necessary examples wanted in our video
# Script for the video


def Video():

    # Demonstration of Data discretization
    # Two methods are used, qcut and cut
    # Qcut puts an equal amount of data in each bin
    # cut puts data in equal sized bins, useful when lots of 0s or repeat values in data

    current = Nt.NaiveBayes('Data/unbinned/glass.data')
    current.bin(4, Print=True)


    # This portion of the code demonstrates a sample trained model.
    # Demonstrates the counting process for class conditional attributes.
    # Demonstrates counting process for a class

    input("Press Enter to continue...")
    current = Nt.NaiveBayes('Data/house-votes-84.data')
    fold1 = current.df.groupby('class', group_keys=False).apply(lambda x: x.sample(frac=.5))
    fold2 = current.df.drop(fold1.index)
    trainP = current.train(fold1, True)
    results = current.test(fold2, trainP)


    # This portion shows the sample outputs from non-shuffled compared to shuffled
    # Non-Shuffled Data

    input("Press Enter to continue...")
    current = Nt.NaiveBayes('Data/iris.data')
    fold1 = current.df.groupby('class', group_keys=False).apply(lambda x: x.sample(frac=.5))
    fold2 = current.df.drop(fold1.index)
    trainP = current.train(fold1, False)
    results = current.test(fold2, trainP, Print=True)


    # Shuffled data set

    input("Press Enter to continue...")
    fold1 = current.df.groupby('class', group_keys=False).apply(lambda x: x.sample(frac=.5))
    fold2 = current.df.drop(fold1.index)
    trainP = current.train(fold1, False)
    results = current.test(fold2, trainP, Print=True)

    # Performance based on precision and recall

    input("Press Enter to continue...")
    file = 'iris.data'
    Recall, Precision = run(file)
    Pcontrol = Precision
    Rcontrol = Recall
    Recall, Precision = runS(file)
    Pshuffled = Precision
    Rshuffled = Recall

    print(file, ":", 'Control Precision:', Pcontrol, 'Shuffled Precision:', Pshuffled, 'Control Recall:', Rcontrol, 'Shuffled Recall:', Rshuffled)

allFiles()
#Video()