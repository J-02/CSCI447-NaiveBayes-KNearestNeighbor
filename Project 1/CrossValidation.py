import matplotlib.pyplot as plt
import numpy as np

import Naivetest as Nt
import DataScrambler as ds
import os

# todo: 5x2 Cross Validation, run 2 fold 5 times, run test 5 times find most accurate
# tests files that end with scrambled vs those that dont
# file.endswith('scrambled.data')



# Goes through files from /Data/ folder to find .data folders
# once found initializes the file with Naive Bayes class
# Creates training data using dataframe sample
# Creates testing data by dropping testing data from original data


def run(file):
    current = Nt.NaiveBayes('Data/' + file)
    if file.startswith('glass'):
        current.bin(8)
    if file.startswith('iris'):
        current.bin(6)
    trainData = current.df.groupby('class', group_keys=False).apply(lambda x: x.sample(frac=.5))
    testData = current.df.drop(trainData.index)
    trainP = current.train(trainData)
    accuracy = current.test(testData, trainP)
    print(file,': ')
    print('Loss:',accuracy[0], 'Precision:',accuracy[1])
    return accuracy



ds.scramble()
scrambled = []
control = []
files = []
for file in os.listdir("Data"):
    if file.endswith('.data'):
        lp = run(file)
        if file.endswith('scrambled.data'):
            scrambled.append(lp[1])
        else:
            files.append(file[0:-5])
            control.append(lp[1])
c = np.subtract(control,scrambled)
plt.plot(files,c)
plt.show()

