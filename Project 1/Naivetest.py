import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NaiveBayes:

    # Imports file path give to pandas dataframe
    # Identifies unique classes in the data set
    # Identifies columns / features in the data set

    def __init__(self, path):
        self.name = path
        self.df = pd.read_csv(path, index_col=0, header=0)
        self.classes = self.df['class'].unique()
        self.features = self.df.columns[self.df.columns != 'class']

    # class_proportion counts total occurrences of a class
    # then calculates a proportion given a data set
    # Calculates Q(C = ci)

    def class_proportion(self, data, c):
        count = data['class'].value_counts()[c]  # total occurrences of class
        total = len(data)
        return count / total

    # 0-1 Loss function, returns 1 if prediction is incorrect returns 0 if true
    def loss(self, row, Class):
        if (row['class'] == Class):
            return 0
        return 1

    # Given a feature (A_j) and value for that feature (a_k) and a class (c_i) calculate probability it is that class
    # Selects a_k in column / feature A_j then selects from those the ones that match the class given: c_j
    # Calculates F(Aj = ak, C = ci)

    def MAP(self, trainData, A_j, a_k, c_i):
        x = trainData.loc[trainData[A_j] == a_k]  # Finds rows with the given feature value
        y = x.loc[trainData['class'] == c_i]  # Finds rows matching class that have the given feature and value
        z = trainData[trainData['class'] == c_i]  # Finds only rows containing the given class
        c_iCount = len(z)
        c_iANDa_k = len(y)
        probability = ((c_iANDa_k + 1) / (c_iCount + len(self.features)))
        return probability

    # Iterates through starting with classes then features and feature values to find probabilities
    # Puts probabilities in a dataframe mapped to a unique ID which is the Feature/Value/Class
    # Also creates another dataframe with a proportion mapped to each unique class in the training set
    # Both dataframes are returned in a list

    def train(self, trainData):
        probabilities = []
        classProportions = []

        for c in trainData['class'].unique():

            for feature in self.features:
                values = trainData[feature].unique()

                for value in values:
                    ID = str(feature) + "," + str(value) + ',' + str(c)
                    probabilities.append([ID, self.MAP(trainData, feature, value, c)])

            classProportions.append([c, self.class_proportion(trainData, c)])

        train = pd.DataFrame(probabilities, columns=['ID', 'Y'])
        classP = pd.DataFrame(classProportions, columns=['Class', 'P'])
        train = train.set_index('ID')
        classP = classP.set_index('Class')
        return [train, classP]

    # Test starts with a row and calculates the probability of each class given the probabilities in the trainData
    # For each class it goes through feature and multiplies the probabilities
    # Once a total probability is found for a class, it compares that to the LargestProb
    # if a larger probability is found it becomes the new LargestProb and its class becomes the predicted class
    # Once all classes are gone through, it checks if its prediction is correct and adds it to the accuracy calculation
    # Accuracy is returned
    # Calculates C(x) and class(x)

    def test(self, testData, trainData):  # x is data frame of probabilities
        # hypothesis = class
        actual = []
        predicted = []
        l = 0
        count = len(testData)
        x = trainData[0]
        classP = trainData[1]

        for index, row in testData.iterrows():

            Class = None
            LargestProb = 0  # prob of most likely class

            for c in self.classes:
                cProb = classP.loc[c, 'P']
                N = len(testData)
                Tprob = cProb / N  # total prob for class
                for feature in self.features:

                    Fprob = 1
                    Id = str(feature) + "," + str(row[feature]) + ',' + str(c)

                    if x.index.__contains__(Id):
                        Fprob = x.loc[Id, 'Y']

                    Tprob *= Fprob
                if (Tprob > LargestProb):
                    LargestProb = Tprob
                    Class = c
            actual.append(row['class'])
            predicted.append((Class))
            l += self.loss(row, Class)

        confusionMatrix = self.confusionMatrix(actual, predicted)
        p = self.Pmacro(confusionMatrix)
        return l, p

    # Binning puts float64 data into bins/categories using pandas qcut and cut
    # when data has lots of 0s it is split using cut as qcut cannot separate into equal bins without overlapping edges
    # number of bins is only changeable per data set

    def bin(self, bins):

        Nbins = bins  # hyperparameter, tunable

        for feature in self.features:
            bin_labels = np.arange(1, Nbins + 1)
            dtype = self.df[feature].dtype
            if dtype == "float64":
                try:
                    self.df[feature] = pd.qcut(self.df[feature], q=Nbins, labels=bin_labels, duplicates='drop')

                except ValueError:
                    self.df[feature] = pd.cut(x=self.df[feature], bins=Nbins, labels=bin_labels)

# todo:not working
# Bin hyperparameter tuning:
# iterates through bin sizes and returns highest accuracy and bin size
# only glass and iris need tuning all others are not binned
# *to run quickly set range(1) and i =>2*
    def tuning(self,equal,file):
        accuracy = []
        nBins = []
        for n in range(2,25):
            i = n
            self.bin(n,equal)
            current = NaiveBayes('Data/' + file)
            trainData = current.df.sample(frac=.50, random_state=1)
            testData = current.df.drop(trainData.index)
            trainP = current.train(trainData)
            test = current.test(testData, trainP)
            accuracy.append(test)
            nBins.append(n)
        plt.scatter(nBins,accuracy)
        plt.show()

# Creates confusion matrix that is used for finding precision loss function
    def confusionMatrix(self,Y,Y_h):
        actual = pd.Series(Y, name='Actual')
        predicted = pd.Series(Y_h, name='Predicted')
        m = pd.crosstab(actual, predicted)
        for x in self.classes:
            if m.columns.__contains__(int(x)):
                pass
            else:
                m.insert(loc=0,column=x, value=0)
        return m

# Find Macro-Averaging precision by adding up precision for each class
    def Pmacro(self, m):
        precision = 0
        for x in self.classes:
            Tp = m.loc[x,x]
            Fp = m[x].sum()-Tp
            if (Tp+Fp) == 0:
                pass
            else:
                p = (Tp/(Tp+Fp))
                precision += p
        pmacro = precision / len(self.classes)
        return pmacro





