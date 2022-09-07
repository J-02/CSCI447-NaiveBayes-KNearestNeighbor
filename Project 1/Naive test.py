import numpy as np
import pandas as pd
import os
    #todo: 10 fold cross validation

    # NaiveBayes class initializes with the path of the .data file and creates a data frame then locates classes and features

class NaiveBayes:
    def __init__(self, path):
        self.name = path
        self.df = pd.read_csv(path, index_col=0,header=0) # imports .data to dataframe
        self.classes = self.df['class'].unique() # creates list of classes / Y / outputs
        self.features = self.df.columns[self.df.columns != 'class']

    # get_classes iterates through classes and appends the proportion of classes to a list

    def get_classes(self):
        No_C = []
        for n in self.classes:  # iterates through classes and appends proportion to the list
            count = self.df['class'].value_counts()[n] # total occurrences of class
            No_C.append([n, count/self.df.shape[0],count]) # append to list
        return No_C

    # Given a feature (A_j) and value for that feature (a_k) and a class (c_i) calculate probability it is that class
    # Selects a_k in column / feature A_j then selects from those the ones that match the class given: c_j

    def MAP(self, A_j, a_k, c_i):
        x = self.df.loc[self.df[A_j] == a_k]
        y = x.loc[self.df['class'] == c_i]
        a_kCount = len(x)
        c_iANDa_k = len(y)
        probability = c_iANDa_k / a_kCount
        print(self.name+": "+"Feature:",A_j,"Value:",a_k,"Class:",c_i,"Probability: ",probability)
        return [A_j,a_k,c_i,probability]

    # Iterates through starting with classes then features and feature values to find probabilities
    # todo: capture all the information and make predictions with it

    def train(self):
        self.bin(5)
        for c in self.classes:
            for feature in self.features:
                values = self.df[feature].unique()
                for value in values:
                    self.MAP(feature,value,c)

    # Binning puts float64 data into bins/categories using pandas qcut and cut
    # when data has lots of 0s it is split using cut as qcut cannot separate into equal bins without overlapping edges
    # number of bins is only changeable per data set
    # todo: optimization function

    def bin(self, bins):
        Nbins = bins # hyperparameter, tunable

        for feature in self.features:
            bin_labels = np.arange(1, Nbins + 1)
            dtype = str(self.df[feature].dtype)
            if dtype == "float64":
                try:
                    self.df[feature] = pd.qcut(self.df[feature],q = Nbins,labels=bin_labels,duplicates='drop')

                except ValueError:
                    self.df[feature] = pd.cut(x=self.df[feature], bins=Nbins, labels=bin_labels)


# Goes through files from /Data/ folder to find .data folders
# once found initializes the file with Naive Bayes class
# adds classes and class proportions to list along with the file name

files = [] # list of .data files
for file in os.listdir("Data"):
    if file.endswith('.data') and not file.endswith('scrambled.data'): # avoids scrambled files for now
        cF = NaiveBayes('Data/'+file)
        files.append([file,cF.get_classes()]) # adds classes and classes counts to list
        print(cF.name)
        cF.train()