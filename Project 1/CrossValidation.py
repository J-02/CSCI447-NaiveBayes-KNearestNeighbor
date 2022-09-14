import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Naivetest as Nt
import DataScrambler as ds
import os
from tqdm import tqdm

# Cross Validation runs test 5 times find mean of precision
def crossValidation(file):
    avg = []
    for i in range(5):
        avg.append(run(file)[1])
    return np.average(avg)

# Creates training data using dataframe sample
# Creates testing data by dropping testing data from original data

def run(file):
    current = Nt.NaiveBayes('Data/' + file)
    trainData = current.df.groupby('class', group_keys=False).apply(lambda x: x.sample(frac=.5))
    testData = current.df.drop(trainData.index)
    trainP = current.train(trainData)
    accuracy = current.test(testData, trainP)
    #print(file,': ')
    #print('Recall:',accuracy[0], 'Precision:',accuracy[1])
    return accuracy

# Goes through files from /Data/ folder to find .data folders
# once found performs cross validation
def main():
    ds.scramble()
    scrambled = []
    control = []
    files = []
    for file in tqdm(os.listdir("Data"), desc="Loading..."):
        if file.endswith('.data'):
            avgPrecision = crossValidation(file)
            if file.endswith('scrambled.data'):
                scrambled.append(avgPrecision)
            else:
                files.append(file[0:-5])
                control.append(avgPrecision)
    difference = np.subtract(control,scrambled)
    data = list(zip(control,scrambled,difference))
    results = pd.DataFrame(data,index=files, columns=['control','scrambled','difference'])
    results.to_csv('Results/Results.data')
    plot(difference, files)

# plots results in 2 line charts
def plot(difference, files):
    plt.plot(files, difference)
    plt.savefig('Results/Results.png')

main()