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
    if (file.startswith(('glass', 'iris'))):
        current.bin(8)
    trainData = current.df.groupby('class', group_keys=False).apply(lambda x: x.sample(frac=.5))
    testData = current.df.drop(trainData.index)
    trainP = current.train(trainData)
    accuracy = current.test(testData, trainP)
    print(file,':',accuracy)
    return accuracy



ds.scramble()
for file in os.listdir("Data"):
    if file.endswith('.data'):
        run(file)