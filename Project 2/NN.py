import pandas as pd
import numpy as np
import CrossValidation as cv
import DistanceFunctions as dist
from itertools import repeat


# Nearest Neighbor
# -----------------------
# Will add all nearest neighbor algorithms here
# todo: k-means, regression
class NearestNeighbor:

    # initializes training data and determines the types of

    def __init__(self, data):
        self.data = pd.read_csv("Data/"+data,index_col=0, header=0)
        self.name = data
        if self.name in ['breast-cancer-wisconsin.data', 'soybean-small.data']:
            self.type = 'discrete'
        else:
            self.type = 'other'
        self.samples = cv.getSamples(data)
        self.k = 5
        self.train = pd.read_csv("Data/"+data,index_col=0, header=0).iloc[0:0]
        if 'class' in self.train.columns:
            self.classification = True
        else:
            self.classification = False

    def getDistance(self, x, y,p={}):
        #print('getting distance:', x, y)
        if not isinstance(x, pd.DataFrame):
            X = pd.DataFrame(self.data.loc[x]).transpose()
        if not isinstance(y, pd.DataFrame):
            Y = pd.DataFrame(self.data.loc[y]).transpose()

        if self.type == 'discrete':
        #needs work, currently outputs numpy array todo
            return y, dist.VDM(self.train, X, Y,p), Y.iat[0,-1]

        else:
            return y, dist.EuclideanD(X,Y), Y.iat[0,-1]

    def getNieghbors(self, point,p={}):
        y = self.train.to_dict('index')
        distance = [self.getDistance(point, key,p) for key in self.train.to_dict('index').keys()]
        distance.sort(key=lambda x: x[1])
        return distance[:self.k]

    def predict(self, x,p={}):
        neighbors = pd.DataFrame(self.getNieghbors(x,p))

        if self.classification:
            Px = neighbors.iloc[:,-1].value_counts().idxmax()
            #print('prediction:',Px)
            return Px
        else:
            pass

    def KNN(self):
        self.train = pd.concat((self.samples[0:9]))
        test = self.samples[9]
        p = dist.initialize(self.train)
        performance = []
        len = test.shape[0]
        for key in test.to_dict('index').keys():
            px = self.predict(key,p)
            actual = test.loc[key,]["class"]
            outcome = self.correct(px, actual)
            #print("actual:", actual)
            #print('remaining:',len)
            performance.append(outcome)
            len -= 1
        print(performance.count(True),"/",test.shape[0])
        return performance.count(True)

# Removes incorrectly classified points until performance degrades todo: add terminal conditions

    def EKNN(self):
        self.train = pd.concat((self.samples[0:9]))
        len = self.train.shape[0]
        performance = 0
        tested = []
        while True:

            x = self.train.sample(n = 1)
            if x.index not in tested:
                tested.append(x.index)
                self.train.drop(x.index)
                if self.correct(self.predict(x), int(x.iloc[0,-1:])):
                    self.train.append(x)

                print(tested.__len__())
                if tested.__len__() == len:
                    print(performance)
                    break
                if tested.__len__() % 50 == 0:
                    test = self.test1()
                    if performance > test:
                        print(performance)
                        break
                    performance = test

# todo: returns performance based on output type on current data
    def test1(self):
        test = self.samples[9]
        performance = [self.correct(self.predict(key), test.loc[key].iat[-1]) for key in test.to_dict('index').keys()]
        return performance.count(True)


    def correct(self, estimate, actual, eps=0):

        if self.classification:
            if estimate == actual:
                return True
            else:
                return False
        else:
            if abs(estimate-actual) < eps:
                return True
            else:
                return False




test = NearestNeighbor('breast-cancer-wisconsin.data')

print(test.KNN())




