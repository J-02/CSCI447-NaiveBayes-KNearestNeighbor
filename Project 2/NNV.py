import pandas as pd
import numpy as np
import CrossValidation as cv
import DistanceV as dist
from tqdm.auto import tqdm, trange
from math import exp
import os
from functools import wraps
import time
def timeit(my_func):
    @wraps(my_func)
    def timed(*args, **kw):
        tstart = time.time()
        output = my_func(*args, **kw)
        tend = time.time()

        print('"{}" took {:.3f} ms to execute\n'.format(my_func.__name__, (tend - tstart) * 1000))
        return output

    return timed

# Nearest Neighbor
# -----------------------
# add all nearest neighbor algorithms here
# todo: k-means, regression, epsilon
class NearestNeighbor:

    # initializes training data and determines the types of data and the type of output
    def __init__(self, data):
        self.data = pd.read_csv("Data/" + data, index_col=0, header=0)  # Creates data frame of all data in dataset
        self.name = data  # saves dataset name
        if self.name in ['breast-cancer-wisconsin.data', 'soybean-small.data']:  # checks if values are discrete or
            # continuous
            self.discrete = True  # uses VDM distance function
        else:
            self.discrete = False  # uses Euclidean distance function
        self.samples, self.tune = cv.getSamples(data)  # creates 10 stratified samples from the dataset in a list [sample1,...,
        # sample10]
        self.k = 5  # how many nearest neighbors to use
        self.train = pd.read_csv("Data/" + data, index_col=0, header=0).iloc[
                     0:0]  # creates empty dataframe with headers
        if 'class' in self.train.columns:  # checks if the dataset is classification or regression
            self.classification = True  # classification
            self.result = "Accuracy"
        else:
            self.classification = False  # regression
            self.eps = 0.5  # todo epsilon needs tuned
            self.bandwith = 100 # todo bandwidth needs tuned
            self.result = "MSE"

    # predict
    # -------
    # decides how to make prediction based on output type
    def predict(self, neighbors):
        if self.classification:
            px = self.classify(neighbors)
        else:
            px = self.regression(neighbors)
        return px

    def EKNN(self):
        # tune epsilon to determine if correct results are correct
        self.train = pd.concat((self.samples[0:9]))  # last sample is test data
        train = self.train
        len = self.train.shape[0]
        performance = 0
        count = 0

        trainV = train.to_numpy()
        performance = []

        if not self.discrete:
            for x in train.to_dict('index').keys():
                if count == len:
                    break
                count += 1
                idf = train.loc[x,:]
                train = train.drop(idf.name)
                i = idf.to_numpy()
                trainV = train.to_numpy()
                distance = np.sum((i - trainV) ** 2, axis=1) ** (1 / 2)
                train['Dist'] = distance
                neighbors = train.nsmallest(self.k, 'Dist').iloc[:, -2:].values
                train.drop('Dist',axis=1,inplace=True)
                px = self.predict(neighbors)
                if self.correct(px, i[-1]):
                    i = np.transpose(i[:,None])
                    train = pd.concat([idf,train])


        else:
            t, p = dist.initialize(train)
            testD = test.to_dict('index')
            for x in testD.values():
                distance = dist.VDM(t, x, p)
                train['Dist'] = distance
                neighbors = train.nsmallest(5, 'Dist').iloc[:, -2:].values
                px = self.predict(neighbors)
                self.correct(px, list(x.values())[-1])

    # KNN
    # ---
    # calculates performance of nearest neighbor algorithm based on training data
    # for regression eps should be 0
    def KNN(self, tune=False, editing=False):
        if tune:
            test = self.tune
            self.train = pd.concat((self.samples))  # combines all but one sample
        else:
            test = self.samples[9]
            self.train = pd.concat((self.samples[0:9]))# last sample is test data

        train = self.train
        trainV = train.to_numpy()
        testV = test.to_numpy()
        performance = []


        if not self.discrete:
            for i in testV:
                distance = np.sum((i-trainV)**2,axis=1)**(1/2)
                train['Dist'] = distance
                neighbors = train.nsmallest(self.k, 'Dist').iloc[:,-2:].values
                px = self.predict(neighbors)
                performance.append(self.correct(px, i[-1]))

        else:
            t, p = dist.initialize(train)
            testD = test.to_dict('index')
            for x in testD.values():
                distance = dist.VDM(t, x, p)
                train['Dist'] = distance
                neighbors = train.nsmallest(5, 'Dist').iloc[:,-2:].values
                px = self.predict(neighbors)
                performance.append(self.correct(px, list(x.values())[-1]))

        correct = self.evaluate(performance) / testV.shape[0]

        return correct

    # correct
    # ------------------------
    # prediction: the predicted class or value in the case of regression
    # actual: actual class or value
    # eps: epsilon, error allowed for regression to be considered correct
    def correct(self, prediction, actual):

        if self.classification:
            if prediction == actual:
                return True
            else:
                return False
        else:
            return abs(prediction-actual)**2

    # evaluate
    # --------
    # returns performance based on output type
    def evaluate(self, performance):
        if self.classification:
            return performance.count(True)
        else:
            return sum(performance)

    # tuneK
    #--------------------------------------
    # tunes k for the data set to use with KNN EKNN and Kmeans
    # sets the k for the data set, carries through all functions
    @timeit
    def tuneK(self):
        k = 1
        tune = {}
        for i in trange(100):

            self.k = k
            performance = self.KNN(tune=True)[0]
            tune[k] = performance
            if performance == 1:
                break
            k += 2

        if self.classification:
            kk = max(tune, key=tune.get)

        else:
            kk = min(tune, key=tune.get)

        self.k = kk

        return kk

    # classify
    # --------
    # gets class with most votes
    def classify(self, neighbors):

        neighbors = pd.DataFrame(neighbors)
        Px = neighbors.iloc[:,0].value_counts().idxmax()

        return Px

    # regression
    # ----------
    # gets predicted output using gaussian kernel
    def regression(self, kN):

        h = self.bandwith # 100 for machines 1 works for abalone
        numer = sum([self.gaussianK(i[1]/h)*i[0] for i in kN])
        denom = sum([self.gaussianK(i[1]/h) for i in kN])
        px = (numer / denom) + self.eps
        #print('Prediction:', px)

        return px

    @staticmethod
    def gaussianK(u):
        x = (-u**2) / 2
        x = exp(x)
        x = x / (2*np.pi)**(1/2)
        return x

    # tuneEpsilon
    # -------------
    # used for regression
    # tunes epsilon to minimize MSE of the data set

    def tuneEpsilon(self):
        pass

    # tuneBandwidth
    # -------------
    # tunes the bandwidth for the gaussian kernel used in regression
    # tuned by minimizing MSE for the data set

    def tuneBandwidth(self):
        pass

    # tune
    # ----
    # tunes all hyperparameters for the data set
    # if classification only tunes K
    # if classification tunes k then performs grid search of bandwidth and epsilon
    def tune(self):
        pass


for file in os.listdir("Data"):
    if file.endswith('.data'):
        print(file)
        test = NearestNeighbor(file)
        print(test.EKNN())
        print(test.result,": ", test.KNN())

