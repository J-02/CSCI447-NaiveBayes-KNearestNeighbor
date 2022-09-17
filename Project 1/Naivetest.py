import numpy as np
import pandas as pd


class NaiveBayes:

    # Imports file path give to pandas dataframe
    # Identifies unique classes in the data set
    # Identifies columns / features in the data set

    def __init__(self, path):
        self.name = path
        self.df = pd.read_csv(path, index_col=0, header=0)
        self.classes = self.df['class'].unique()
        self.features = self.df.columns[self.df.columns != 'class']

    # Training data:
    # -------------------------------------------------------------
    # train
    # Iterates through starting with classes then features and feature values to find probabilities
    # Puts probabilities in a dataframe mapped the Feature/Value/Class
    # Also creates another dataframe with a proportion mapped to each unique class in the training set
    # Both dataframes are returned in a list

    def train(self, trainData, Print=False):
        probabilities = []
        counts = []
        classProportions = []
        classCounts = []

        for c in trainData['class'].unique():

            for feature in self.features:
                values = trainData[feature].unique()

                for value in values:
                    ID = str(feature) + ", " + str(value) + ', ' + str(c)
                    probability, count = self.Parameterization(trainData, feature, value, c)
                    probabilities.append([ID, probability])
                    counts.append([ID, count])
            classCount, total = self.class_proportion(trainData, c)
            classProportions.append([c, classCount / total])
            classCounts.append([c, classCount])
        train = pd.DataFrame(probabilities, columns=['Attribute, Value, Class', 'Probability'])
        count = pd.DataFrame(counts, columns=['Attribute, Value, Class', 'Occurrences'])
        classP = pd.DataFrame(classProportions, columns=['Class', 'Proportion'])
        classC = pd.DataFrame(classCounts, columns=['Class', 'Count'])
        train = train.set_index('Attribute, Value, Class')
        count = count.set_index('Attribute, Value, Class')
        classP = classP.set_index('Class')
        classC = classC.set_index('Class')

        # This is for the video

        if Print:
            print("This is a trained model")
            print(train)
            print("This shows the counts for class-conditional attributes")
            print(count)
            print("This shows the proportion of each class")
            print(classP)
            print("This shows the count for each class")
            print(classC)

        return [train, classP]

    # class_proportion counts total occurrences of a class
    # then calculates a proportion given a data set
    # Calculates Q(C = ci)

    def class_proportion(self, data, c):
        count = data['class'].value_counts()[c]  # total occurrences of class
        total = len(data)
        return count, total

    # Parameterization
    # Given a feature (A_j) and value for that feature (a_k) and a class (c_i) calculates the probability it is c_i
    # Selects a_k from A_j (column / feature) then selects from those the ones that match the class given: c_j
    # Calculates F(A_j = a_k, C = c_i)

    def Parameterization(self, trainData, A_j, a_k, c_i):
        x = trainData.loc[trainData[A_j] == a_k]  # Finds rows with the given feature value
        y = x.loc[trainData['class'] == c_i]  # Finds rows matching class that have the given feature and value
        z = trainData[trainData['class'] == c_i]  # Finds only rows containing the given class
        c_iCount = len(z)
        c_iANDa_k = len(y)
        probability = ((c_iANDa_k + 1) / (c_iCount + len(self.features)))
        count = c_iANDa_k
        return probability, count

    # Testing data
    # ---------------------------------------
    # test
    # Test starts with a row and calculates the probability of each class given the probabilities in the trainData
    # For each class it goes through features of the row and multiplies the probabilities
    # Once a total probability is found for a class, it compares that to the LargestProb
    # if a larger probability is found it becomes the new LargestProb and its class becomes the predicted class
    # Once all classes are gone through, it checks if its prediction is correct and adds it to the accuracy calculation
    # recall and precision are returned
    # Calculates C(x) and class(x)

    def test(self, testData, trainData, Print=False):

        testData.reset_index()
        actual = []
        predicted = []
        x = 0
        trainP = trainData[0]  # data frame of probabilities given attribute, attribute value, and class
        classP = trainData[1]  # data frame of class probabilities
        for index, row in testData.iterrows():

            Class = None
            LargestProb = 0  # prob of most likely class

            for c in self.classes:
                cProb = classP.loc[c, 'Proportion']
                N = len(testData)
                Tprob = cProb / N  # total prob for class
                for feature in self.features:

                    Fprob = 1
                    Id = str(feature) + ", " + str(row[feature]) + ', ' + str(c)

                    if trainP.index.__contains__(Id):
                        Fprob = trainP.loc[Id, 'Probability']  # probability from parameterizing the training data

                    Tprob *= Fprob

                if Tprob > LargestProb:
                    LargestProb = Tprob
                    Class = c
            actual.append(row['class'])
            predicted.append(Class)

            if Print:
                if row['class'] != Class:
                    y = 'Incorrect'
                else:
                    y = 'Correct'
                print(y, testData.iloc[[x], 0:4], "Predicted Class:", Class, "Actual:", row['class'])
            x += 1

        confusionMatrix = self.confusionMatrix(actual, predicted)
        p = self.Pmacro(confusionMatrix)
        r = self.Rmacro(confusionMatrix)
        if Print: testData.insert(0, "Predicted Class", predicted)
        return [r, p]

    # Data discretization:
    # -----------------------------------------------------------------------------------
    # bin
    # bin puts float64 data into bins/categories using pandas qcut and cut
    # when data has lots of 0s it is split using cut as qcut cannot separate into equal bins without overlapping edges
    # number of bins is only changeable per data set

    def bin(self, bins, Print=False):

        Nbins = bins  # hyperparameter, tunable

        for feature in self.features:
            bin_labels = np.arange(1, Nbins + 1)
            dtype = self.df[feature].dtype
            if dtype == "float64":
                try:
                    if Print:

                        print(pd.qcut(self.df[feature], q=Nbins).value_counts())
                        print('qcut')
                    self.df[feature] = pd.qcut(self.df[feature], q=Nbins, labels=bin_labels)

                except ValueError:
                    if Print:
                        print(pd.cut(self.df[feature], bins=Nbins).value_counts())
                        print('cut')
                    self.df[feature] = pd.cut(x=self.df[feature], bins=Nbins, labels=bin_labels)

    # Loss functions:
    # ---------------------------------------------------------------------
    # confusionMatrix
    # Creates confusion matrix that is used for finding loss functions
    def confusionMatrix(self, Y, Y_h):
        actual = pd.Series(Y, name='Actual')
        predicted = pd.Series(Y_h, name='Predicted')
        m = pd.crosstab(actual, predicted)
        for x in self.classes:
            if m.columns.__contains__(x):
                pass
            else:
                m.insert(loc=0, column=x, value=0)
        m.to_csv('Results/ConfusionMatrix.data')
        return m

    # Find Macro-Averaging recall by adding up recall calculation for each class
    def Rmacro(self, m):
        recall = 0
        for x in self.classes:
            Tp = m.loc[x, x]
            Fp = m[x].sum() - Tp
            if (Tp + Fp) == 0:
                pass
            else:
                p = (Tp / (Tp + Fp))
                recall += p
        rmacro = recall / len(self.classes)
        return rmacro

    # Find Macro-Averaging precision by adding up precision calculation for each class
    def Pmacro(self, m):
        precision = 0
        for x in self.classes:
            Tp = m.loc[x, x]
            Fp = m.loc[x].sum() - Tp
            if (Tp + Fp) == 0:
                pass
            else:
                p = (Tp / (Tp + Fp))
                precision += p
        pmacro = precision / len(self.classes)
        return pmacro

    # 0-1 Loss function, returns 1 if prediction is incorrect returns 0 if true, not used
    def loss(self, row, Class):
        if row['class'] == Class:
            return 0
        return 1