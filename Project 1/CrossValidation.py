import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Naivetest as Nt
import DataScrambler as ds
import os
from tqdm import tqdm

# Cross Validation runs test 5 times find mean of precision
def Evaluation(file):
    Ravg = []
    Pavg = []
    for i in range(100):
        Ravg.append(run(file)[0])
        Pavg.append(run(file)[1])
    return [np.average(Ravg),np.average(Pavg)]

# Creates training data using dataframe sample
# Creates testing data by dropping testing data from original data

def run(file):
    ds.scramble()
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
# performs 5X2 cross validation with data files
def Crossvalidation():
    Pcontrol = []
    Rcontrol = []
    Pscrambled = []
    Rscrambled = []
    files = []
    for file in tqdm(os.listdir("Data"), desc="Loading..."):
        if file.endswith('.data'):
            avgPrecision = Evaluation(file)[1]
            avgRecall = Evaluation(file)[0]
            if file.endswith('scrambled.data'):
                Pscrambled.append(avgPrecision)
                Rscrambled.append(avgRecall)
            else:
                files.append(file[0:-5])
                Pcontrol.append(avgPrecision)
                Rcontrol.append(avgRecall)

    Pdifference = np.subtract(Pcontrol, Pscrambled)
    Rdifference = np.subtract(Rcontrol, Rscrambled)
    Pdata = list(zip(Pcontrol,Pscrambled,Pdifference))
    Rdata = list(zip(Rcontrol, Rscrambled, Rdifference))
    Presults = pd.DataFrame(Pdata,index=files, columns=['control','scrambled','difference'])
    Rresults = pd.DataFrame(Rdata, index=files, columns=['control', 'scrambled', 'difference'])
    Presults.to_csv('Results/PrecisionResults.data')
    Rresults.to_csv('Results/RecallResults.data')
    plt.bar(np.arange(len(files)) - 0.125, Pdifference, 0.25, label="Precision")
    plt.bar(np.arange(len(files)) + 0.125, Rdifference, 0.25,label ="Recall")
    plt.title('Performance in Scrambled vs Unscrambled Datasets')
    plt.xticks(np.arange(len(files)), files)
    plt.ylabel("Difference in Performance")
    plt.xlabel("Dataset", fontsize = 8)
    plt.legend()
    plt.savefig('Results/Results.png')


# Does 5x2 CrossValidation for each data set
def CV2():
    Pcontrol = []
    Rcontrol = []
    Pscrambled = []
    Rscrambled = []
    files = []
    a = False
    b = False
    for file in os.listdir("Data"):

        if file.endswith('.data'):
            for x in range(5):
                Recall, Precision = run(file)
                if file.endswith('scrambled.data'):
                    Pscrambled.append(Precision)
                    Rscrambled.append(Recall)
                    a = True
                else:
                    name = file[0:-5]
                    Pcontrol.append(Precision)
                    Rcontrol.append(Recall)
                    b = True
        if (a and b):
            a = False
            b = False

            scrambledP = pd.Series(Pscrambled, name='Scrambled Precision')
            controlP = pd.Series(Pcontrol, name='Control Precision')
            scrambledR = pd.Series(Rscrambled, name='Scrambled Recall')
            controlR = pd.Series(Rcontrol, name='Control Recall')
            data = [controlP,scrambledP,controlR,scrambledR]
            results = pd.DataFrame(data).transpose()
            print(results)
            results.to_latex()
            Pcontrol = []
            Rcontrol = []
            Pscrambled = []
            Rscrambled = []



CV2()