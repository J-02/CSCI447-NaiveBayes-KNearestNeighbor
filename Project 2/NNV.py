import pandas as pd
import numpy as np
import CrossValidation as cv
import DistanceV as dist
from tqdm.auto import tqdm, trange
from math import exp
import os
from functools import wraps
import time
import matplotlib.pylab as plt


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
# todo: k-means,  epsilon
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
        self.train = pd.concat((self.samples[0:9]))  # creates training data dataframe with headers
        if 'class' in self.train.columns:  # checks if the dataset is classification or regression
            self.classification = True  # classification
            self.result = "Accuracy"
        else:
            self.classification = False  # regression
            self.eps = 0.5  # todo epsilon needs tuned
            self.bandwith = 1000 # todo bandwidth needs tuned
            self.result = "MSE"
        self.clusters = 10

    # predict
    # -------
    # decides how to make prediction based on output type
    def predict(self, neighbors):
        if self.classification:
            px = self.classify(neighbors)
        else:
            px = self.regression(neighbors)
        return px

    # Eknn
    # ----
    # edited nearest neighbor, removes incorrectly classified points from training data
    def EKNN(self, tune=False):
        # todo: tune epsilon to determine if correct results are correct
        if tune:
            test = self.tune
        else:
            test = self.samples[9]

        # initialization
        train = self.train  # initializes local training set to edit
        edited = train # initializes local edited training set to test performance for reduced data set
        len = self.train.shape[0]  # Creates variable to check if we have gone over every element in the training data
        if self.classification: currentperf = 0  # determines what value performance starts at
        if not self.classification: currentperf = np.infty  # lower is better for regression performance value MSE
        count = 0  # initializes how many times we have edited
        performance = []  # initializes list of performance values
        increment = (len // 100)  # how often to test performance
        prevSet = edited  # initializes variable to store previous edited set
        currentperf = self.performance(test,train)  # sets initial performance

        # for regression data sets
        if not self.classification:
            for x in (train.to_dict('index').keys()):  # iterating through each element in training data
                if count == len: # if all training data has be gone through
                    if tune:
                        return currentperf

                    return edited  # returns edited training set
                count += 1  # adds to amount of elements gone through in training data

                edited = self.edit(x, edited)

                if edited.shape[0] < prevSet.shape[0]:  # tests if condition is met, for every 20th of the data set gone through
                    correct = self.performance(test, edited)
                    # tests if error increased
                    if correct > currentperf:
                        # returns edited data set with best performance, the performance, and the starting size of training data
                        self.clusters = (len - edited.shape[0])
                        if tune:
                            return currentperf
                        return edited, currentperf, self.clusters, performance
                    else:
                        # updates performance if it improved and saves the edited training data
                        currentperf = correct
                        prevSet = edited
                        performance.append(correct)

        # for classification
        else:

            for x in tqdm(train.to_dict('index').keys()):
                count += 1
                train = self.edit(x, edited)  # returns data set with or without x depending if correct
                if count % increment == 0 and count != 0:  # tests if condition is met, for every 20th of the data set gone through
                    correct = self.performance(test, edited)
                    # tests if error increased
                    if correct < currentperf:
                        # returns edited data set with best performance, the performance, and the starting size of training data
                        if tune:
                            return currentperf
                        return edited, currentperf, len, performance
                    else:
                        # updates performance if it improved and saves the edited training data
                        currentperf = correct
                        prevSet = edited
                        performance.append(correct)
        if tune:
            return currentperf

    # edit
    # ----
    # gets prediction for element being edited
    # returns edited dataframe
    # element removed if incorrect
    def edit(self, x, train):

        idf = train.loc[x, :]
        x = train.loc[x, :].to_dict()  # picks singular element from training data
        train = train.drop(index=idf.name)  # drops that element from training data
        i = np.array(idf.iloc[:-1].to_numpy() ) # sets element to array for optimized calculation and removes class data
        trainV = np.array(train.iloc[:,:-1].to_numpy()) # sets training data to array for optimized calculation and removes class data

        if self.discrete: t, p = dist.initialize(train)  # initializes VDM for each element in training data
        if self.discrete:  # uses VDM if discrete variables
            distance = dist.VDM(t, x, p)
        else:  # uses Euclidean in continuous
            distance = np.sum((i - trainV) ** 2, axis=1) ** (1 / 2)  # calculates distance to all training elements from test element

        train['Dist'] = distance  # adds distance of each element to new data frame column, to determine classes
        neighbors = train.nsmallest(self.k, 'Dist').iloc[:, -2:].values  # finds the k-nearest neighbors
        train.drop('Dist', axis=1, inplace=True)  # removes distance column
        px = self.predict(neighbors)
        # for regression data sets
        if not self.classification:
            error = abs(px - i[-1])
            allowederror = self.eps*i[-1]
            if error > allowederror:  # if regression estimate is within epsilon of actual value
                train = pd.concat([pd.DataFrame(idf).transpose(), train])  # readds element to training data if correct
        # for classification data sets
        else:
            if self.correct(px, i[-1]):  # if properly classified added back to training data
                train = pd.concat([pd.DataFrame(idf).transpose(), train])
        return train

    # performance
    # -------
    # tests performance of Eknn
    def performance(self, test, train):
        # initializes testing and training vectors and list to capture performance of each element
        testV = np.array(test.to_numpy())
        trainV = np.array(train.to_numpy())
        tPerf = []
        # goes through all elements of test data
        if self.discrete:
            t, p = dist.initialize(train)  # initializes VDM for each element in training data
            testD = test.to_dict('index')
            for x in testD.values():
                distance = dist.VDM(t, x, p)
                train['Dist'] = distance
                neighbors = train.nsmallest(self.k, 'Dist').iloc[:, -2:].values
                train.drop('Dist', axis=1, inplace=True)
                px = self.predict(neighbors)
                tPerf.append(self.correct(px, list(x.values())[-1]))  # returns squared error / amount correct
            correct = self.evaluate(tPerf) / testV.shape[0]  # returns mean squared error / % correct
        else:
            for x in testV:
                distance = np.sum((x - trainV) ** 2, axis=1) ** (1 / 2)  # calculates distance to all training elements from test element
                train['Dist'] = distance
                neighbors = train.nsmallest(self.k, 'Dist').iloc[:, -2:].values
                train.drop('Dist', axis=1, inplace=True)
                px = self.predict(neighbors)
                tPerf.append(self.correct(px, x[-1]))  # returns squared error / amount correct
            correct = self.evaluate(tPerf) / testV.shape[0]  # returns mean squared error / % correct
        return correct

    # KNN
    # ---
    # calculates performance of nearest neighbor algorithm based on training data
    # for regression eps should be 0
    def KNN(self, tune=False):
        if tune:
            test = self.tune
        else:
            test = self.samples[9]

        train = self.train

        trainV = np.array(train.to_numpy())
        testV = np.array(test.to_numpy())
        performance = []


        if not self.discrete:
            for i in testV:

                distance = np.sum((i-trainV)**2,axis=1)**(1/2)
                train['Dist'] = distance
                neighbors = train.nsmallest(self.k, 'Dist').iloc[:,-2:].values
                train.drop('Dist', axis=1, inplace=True)
                px = self.predict(neighbors)
                performance.append(self.correct(px, i[-1]))

        else:
            t, p = dist.initialize(train)
            testD = test.to_dict('index')
            for x in testD.values():
                distance = dist.VDM(t, x, p)
                train['Dist'] = distance
                neighbors = train.nsmallest(5, 'Dist').iloc[:,-2:].values
                train.drop('Dist', axis=1, inplace=True)
                px = self.predict(neighbors)
                performance.append(self.correct(px, list(x.values())[-1]))

        correct = self.evaluate(performance) / testV.shape[0]

        return correct

    def Kmeans(self):
        # initialize random centroids and varaible to check if centroids changed
        train = self.train
        centroids = train.sample(n=self.clusters,replace=False)
        changed = True
        C = np.array(centroids.to_numpy())
        old = np.array(centroids.to_numpy())
        Col= centroids.columns.values
        lastCcounts = []
        # continue until centroids converge
        pbar = tqdm()
        n = 0
        while changed:
            centroids = pd.DataFrame(old,columns=Col)  # creates dataframe for VDM initialization
            Ccounts = [1 for i in range(self.clusters)]  # creates empty list of counts
            if self.discrete:  # decides distance function to use
                trainD = train.to_dict('index')  # training data to dictionary
                t, p = dist.initialize(centroids)   # initializes VDM maps
                for x in tqdm(trainD.values()):  # goes through all data
                    distances = dist.VDM(t, x, p, means=centroids)  # gets distance from each centroid
                    c = np.argmin(distances[:,-1])  # returns closest centroid
                    xi = np.array((list(x.values())))  # creates array from training point
                    C[c] = C[c,:]+xi  # adds training point to centroid
                    Ccounts[c] += 1  # adds count for centoid
                m = np.rint(C / np.array(Ccounts)[:,None])  # Calulates mean point of centroid
                if np.array_equal(m,old):  # Checks if centroid changed
                    changed = False
                    break
                if Ccounts == lastCcounts:
                    changed = False
                    break
                else:
                    old = np.array(m)
                    lastCcounts = list(Ccounts)

            else:
                trainV = np.array(train.to_numpy())
                for i in trainV:
                    distances = np.sum((i - C) ** 2, axis=1) ** (1 / 2)
                    centroids['Dist'] = distances
                    c = np.argmin(np.array(centroids.to_numpy())[:,-1])
                    centroids.drop('Dist', axis=1, inplace=True)
                    C[c] = C[c, :] + i  # adds training point to centroid
                    Ccounts[c] += 1  # adds count for centoid
                m = np.around((C - old) / (np.array(Ccounts)-1)[:, None])
                if np.array_equal(m,old):  # Checks if centroid changed
                    changed = False
                    break
                if Ccounts == lastCcounts:
                    changed = False
                    break
                else:
                    old = np.array(m)
                    lastCcounts = list(Ccounts)
            n += 1
            pbar.update(n)

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
            return abs(prediction-actual)

    # evaluate
    # --------
    # returns performance based on output type
    def evaluate(self, performance):
        if self.classification:
            return performance.count(True)
        else:
            return sum(performance)

    # classify
    # --------
    # gets class with most votes
    def classify(self, neighbors):

        neighbors = pd.DataFrame(neighbors)
        weights = pd.DataFrame(neighbors.groupby(by=[0])[1].sum())
        Px = weights[weights[1] == weights[1].min()].index.values[0]

        return Px

    # regression
    # ----------
    # gets predicted output using gaussian kernel
    def regression(self, kN):

        numer = sum([self.gaussianK(i[1])*i[0] for i in kN])
        denom = sum([self.gaussianK(i[1]) for i in kN])
        px = (numer / denom)
        #print('Prediction:', px)

        return px

    def gaussianK(self, u):
        x = (u**2) / (2*self.bandwith**2)
        x = exp(-x)
        x = x / (2*np.pi)**(1/2)
        return x

    # tune
    # ----
    # tunes all hyperparameters for the data set
    # if classification only tunes K
    # if regression then performs grid search of bandwidth and k and then epsilon
    @timeit
    def tuneit(self):
        if self.classification:
            return "K, performance: "+str(self.tuneK())
        else:
            return "K, Bandwith: "+str(self.tuneBandwidth()), "epsilon: "+str(self.eps)+" MSE: "+str(self.tuneEpsilon())

    # tuneEpsilon
    # -------------
    # used for regression
    # tunes epsilon to minimize MSE of the data set
    def tuneEpsilon(self):
        perf = {}
        for i in tqdm(np.random.randint(0, 100, 25)/100):
            self.eps = i
            #print(i)
            perf[i] = self.EKNN(tune=True)
        low = min(perf, key=perf.get)
        self.eps = low
        lists = sorted(perf.items())
        x, y = zip(*lists)
        plt.plot(x, y)
        plt.xlabel("epsilon")
        plt.ylabel("MSE")
        plt.savefig("tuning/" + "epsilon."+self.name[:-4]+'png')
        plt.clf()
        return perf[low]


    # tuneBandwidth
    # -------------
    # tunes the bandwidth for the gaussian kernel used in regression
    # tuned by minimizing MSE for the data set

    def tuneBandwidth(self):
        if self.name == 'machine.data':
            x = 100
        elif self.name == 'forestfires.data':
            x = 20
        else:
            x = 1
        results = []

        for k in trange(1,10):
            for h in np.random.randint(x, 10*x, 25):
                self.bandwith = h
                MSE = self.KNN(tune=True)
                results.append([k,h,MSE])

        low = min(results, key=lambda x: x[2])

        # For easier access of columns, convert to numpy array
        results = np.array(results)
        # Now we visualize.
        plt.scatter(results[:, 0], results[:, 1], c=results[:, 2], cmap='cool', s=10)
        plt.xlabel('n_neighbors', fontsize=14)
        plt.ylabel('kernel_width', fontsize=14)
        plt.colorbar().set_label('Mean Absolute Error')
        plt.savefig("tuning/" + "kandbandwith."+self.name[:-4]+'png')
        plt.clf()
        #plt.show()

        k,h = low[:-1]

        self.k = k
        self.bandwith = h
        return k, h
        # tuneK
        # --------------------------------------
        # tunes k for the data set to use with KNN EKNN and Kmeans
        # sets the k for the data set, carries through all functions
    def tuneK(self):
        tune = {}
        self.train = pd.concat((self.samples))
        for k in range(1,self.train.shape[0]**(1/2)+5):
            self.k = k
            performance = self.KNN(tune=True)
            tune[k] = performance
            k += 1

        if self.classification:
           kk = max(tune, key=tune.get)

        else:
            kk = min(tune, key=tune.get)

        self.k = kk

        lists = sorted(tune.items())
        x, y = zip(*lists)
        plt.plot(x, y)
        plt.xlabel("K neighbors")
        if self.classification: plt.ylabel("Prob")
        else: plt.ylabel("MSE")
        plt.savefig("ktune"+self.name[:-4]+'png')
        plt.clf()
        return kk, tune[kk]







#for file in os.listdir("Data"):
    #if file.endswith('.data'):
        #print(file)
        #test1 = NearestNeighbor(file)
        #print(test1.Kmeans())
        #print(test1.tuneit())
        #print(test1.KNN())
        #print(test1.EKNN())
        #print(test1.result,": ", test1.KNN())

