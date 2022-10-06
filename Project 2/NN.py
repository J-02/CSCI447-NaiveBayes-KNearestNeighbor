import pandas as pd
import time
import CrossValidation as cv
import DistanceFunctions as dist
from tqdm import tqdm


# Nearest Neighbor
# -----------------------
# Will add all nearest neighbor algorithms here
# todo: k-means, regression
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
        self.samples = cv.getSamples(data)  # creates 10 stratified samples from the dataset in a list [sample1,...,
        # sample10]
        self.k = 5  # how many nearest neighbors to use
        self.train = pd.read_csv("Data/" + data, index_col=0, header=0).iloc[
                     0:0]  # creates empty dataframe with headers
        if 'class' in self.train.columns:  # checks if the dataset is classification or regression
            self.classification = True  # classification
        else:
            self.classification = False # regression
            self.eps = 10  # todo epsilon needs tuned

    # getDistance:
    # ------------------------------------------
    # determines which distance function to use based on dataset: VDM for discrete, Euclidean for continuous.
    # x: is the vector we are classifying
    # y: is the vector we are getting the distance from
    # p: is the initialized probabilities for performing VDM faster, it is empty by default

    def getDistance(self, x, y, p=None):
        # print('getting distance:', x, y)
        # standardizes data to dataframes if they are not, sometimes an index is used rather than a dataframe object
        # probably an issue here works for now
        if p is None:
            p = {}
        if not isinstance(x, pd.DataFrame):
            X = pd.DataFrame(self.data.loc[x]).transpose()
        if not isinstance(y, pd.DataFrame):
            Y = pd.DataFrame(self.data.loc[y]).transpose()

        # picks distance function to use
        if self.type == 'discrete':
            return y, dist.VDM(self.train, X, Y, p), Y.iat[0, -1]
            # returns index of comparison vector (y),the distance, and the class of y [index, distance, class]
        else:
            return y, dist.EuclideanD(X, Y), Y.iat[0, -1]
            # returns index of comparison vector (y),the distance, and output value of y [index, distance, value]

    # getNeighbors
    # -----------------------------
    # gets list of k-nearest neighbors and returns them
    # point: the vector to get the nearest neighbors of.
    # p: initialized probabilities to pass through to vdm distance function if type is discrete default empty

    def getNeighbors(self, point, p=None):
        if p is None:
            p = {}
        y = self.train.to_dict('index')  # Puts dataframe in dictionary to be easier to iterate through

        # puts distance and class of each neighbor in a list
        distance = [self.getDistance(point, key, p) for key in self.train.to_dict('index').keys()]

        distance.sort(key=lambda x: x[1])  # sorts the list by distance so closest are at the top of the list
        return distance[:self.k]  # returns the part of the list that contains k-nearest neighbors

    # predict
    # ---------------------------------
    # gets a prediction of the input vector
    # x: input vector to find prediction of
    # p: initialized probabilities to pass through to vdm distance function if type is discrete, default empty
    def predict(self, x, p=None):

        if p is None:
            p = {}
        nearest = pd.DataFrame(self.getNeighbors(x, p))  # casts getNeighbors to data frame [index, distance, class]
        kneighbors = self.data.loc[list(nearest[0])]  # grabs information of nearest neighbors from data

        # tests whether regression or classification
        if self.classification:
            Px = kneighbors.iloc[:, -1].value_counts().idxmax()  # gets class with most votes / occurrences
            # print('prediction:',Px)
            return Px  # returns prediction as class
        else:  # todo: need to add regression
            pass

    # KNN
    # ------------------------
    # performs k-nearest neighbor using all but one of the samples as training data and last to test

    def KNN(self):
        self.train = pd.concat((self.samples[0:9]))  # combines all but one sample
        test = self.samples[9]  # last sample is test data
        len = test.shape[0]
        if self.type == 'discrete':  # if data is discrete initializes probabilities from training data for vdm distance
            p = dist.initialize(self.train)
            performance = []  # initializes list of outcomes to measure performance
            for key in tqdm(test.to_dict('index').keys()):  # iterates through each index in the training data
                px = self.predict(key, p)  # gets prediction for index / vector
                actual = test.loc[key,]["class"]  # sets actual class for index / vector
                outcome = self.correct(px, actual)  # checks if prediction == class, returns boolean
                #print("actual:", actual)
                #print('remaining:',len)
                performance.append(outcome)  # adds outcome to performance list
                #len -= 1
        else:
            performance = []  # initializes list of outcomes to measure performance
            for key in test.to_dict('index').keys():  # iterates through each index in the training data
                px = self.predict(key)
                actual = test.loc[key,].iat[-1]  # sets actual value for index / vector
                outcome = self.correct(px, actual, self.eps)  # checks if prediction is with in eps of actual, returns boolean
                # print("actual:", actual)
                # print('remaining:',len)
                performance.append(outcome)  # adds outcome to performance list
                # len -= 1
        print(performance.count(True), "/", test.shape[0]) # prints performance
        return performance.count(True) # returns number of correct predictions

    # EKNN - edited k-nearest neighbor (WIP)
    #---------------------------------
    # Removes incorrectly classified points until performance degrades todo: add terminal conditions

    def EKNN(self):
        self.train = pd.concat((self.samples[0:9]))
        len = self.train.shape[0]
        performance = 0
        tested = []
        while True:

            x = self.train.sample(n=1)
            if x.index not in tested:
                tested.append(x.index)
                self.train.drop(x.index)
                if self.correct(self.predict(x), int(x.iloc[0, -1:])):
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

    # correct
    # ------------------------
    # prediction: the predicted class or value in the case of regression
    # actual: actual class or value
    # eps: epsilon, error allowed for regression to be considered correct
    def correct(self, prediction, actual, eps=0):

        if self.classification:
            if prediction == actual:
                return True
            else:
                return False
        else:
            if abs(prediction - actual) < eps:
                return True
            else:
                return False

    # test1
    #--------------------------
    # tests if the performance of EKNN has degraded
    def test1(self):
        test = self.samples[9]  # initializes test using the last sample

        # loops through all indexs / keys in test data and appends the outcome of predicting each vector to list
        performance = [self.correct(self.predict(key), test.loc[key,].iat[-1]) for key in test.to_dict('index').keys()]
        return performance.count(True)  # returns the amount of correct predictions


test = NearestNeighbor('breast-cancer-wisconsin.data')

print(test.KNN())


