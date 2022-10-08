import pandas as pd
import numpy as np
import CrossValidation as cv
import distoptimized as dist
from tqdm.auto import tqdm, trange
from math import exp


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
            self.type = 'discrete'  # uses VDM distance function
        else:
            self.type = 'other'  # uses Euclidean distance function
        self.samples, self.tune = cv.getSamples(data)  # creates 10 stratified samples from the dataset in a list [sample1,...,
        # sample10]
        self.k = 5  # how many nearest neighbors to use
        self.train = pd.read_csv("Data/" + data, index_col=0, header=0).iloc[
                     0:0]  # creates empty dataframe with headers
        if 'class' in self.train.columns:  # checks if the dataset is classification or regression
            self.classification = True  # classification
        else:
            self.classification = False # regression
            self.eps = 0.5  # todo epsilon needs tuned
            self.bandwith = 100 # todo bandwidth needs tuned

    def KNNv2(self, tune=False):
        if tune:
            test = self.tune
            self.train = pd.concat((self.samples))  # combines all but one sample
        else:
            test = self.samples[9]
            self.train = pd.concat((self.samples[0:9]))# last sample is test data
        train = self.train
        trainV = train.to_numpy()
        testV = test.to_numpy()
        if not self.classification:
            for i in tqdm(testV):
                distance = [dist.EuclideanVector(i, y) for y in trainV]
                train['Dist'] = distance
                neighbors = train.nsmallest(self.k, 'Dist').iloc[:,-2:].values
                px = self.regression(i, neighbors)
                #print("Prediction:",px)
                actual = i[-1]
                #print('Actual:',actual)
                error = actual - px
                error = error**2
                MSE =+ error
            MSE = MSE / testV.shape[0]
            return MSE
        else:
            p = dist.initialize(train)
            performance = []
            for x in (testV):
                distance = [dist.VDMv2(train, x, y, p) for y in trainV]
                train['Dist'] = distance
                neighbors = train.nsmallest(5, 'Dist')
                px = self.classify(neighbors)
                actual = x[-1]  # sets actual value for index / vector
                outcome = self.correct(px, actual)
                print("Actual | Predicted")
                print(actual, "|", px)
                performance.append(outcome)
            return performance.count(True) / testV.shape[0]

    # correct
    # ------------------------
    # prediction: the predicted class or value in the case of regression
    # actual: actual class or value
    # eps: epsilon, error allowed for regression to be considered correct
    def correct(self, prediction, actual, ):

        if self.classification:
            if prediction == actual:
                return True
            else:
                return False
        else:
            if abs(prediction - actual) + self.eps:
                return True
            else:
                return False

    # tuneK
    #--------------------------------------
    # tunes k for the data set to use with KNN EKNN and Kmeans
    # sets the k for the data set, carries throught to all funtions
    def tuneK(self):
        k = 1
        tune = {}
        for i in trange(10):
            self.k = k
            performance = self.KNN(tune=True)
            tune[k] = performance
            if performance == 1:
                break
            k += 2
        l = pd.DataFrame(tune, index=['performance']).transpose()
        kk = l[l.performance == l.performance.max()]
        self.k = kk.iat[0,0]
        return tune
    def classify(self, neighbors):
        Px = neighbors['class'].value_counts().idxmax()
        return Px
    def regression(self, x, kN):
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

    def tuneEpsilon(self):
        pass

    def tuneBandwidth(self):
        pass

test = NearestNeighbor('machine.data')

#print(test.tuneK())
print("MSE",test.KNNv2())
test = NearestNeighbor('breast-cancer-wisconsin.data')
print("% correct",test.KNNv2())
#print(test.EKNN())


