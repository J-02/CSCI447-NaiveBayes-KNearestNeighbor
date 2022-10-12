import pandas as pd
import numpy as np
import Stratification as cv
import DistanceV as dist
from math import exp
import matplotlib.pylab as plt


# Nearest Neighbor
# -----------------------
# Class that maintains all the methods and infromation to run nearest neighbor algorithms
#
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
        # sample10] also saves the sample used to test tuning
        self.train = pd.concat((self.samples[0:9]))  # creates training data from the bottom 9 samples
        if 'class' in self.train.columns:  # checks if the dataset is classification or regression
            self.classification = True  # classification
            self.result = "Accuracy"  # how perfromance is measured
        else:
            self.classification = False  # regression
            self.result = "MSE" # how perfromance is measured
        self.clusters = None  # updated when EKNN is run.

    # predict
    # -------
    # decides how to make prediction based on output type
    def predict(self, neighbors):
        if self.classification:
            px = self.classify(neighbors)  # classify based on neighbors
        else:
            px = self.regression(neighbors)  # gets regression estimate
        return px

    # Eknn
    # ----
    # edited nearest neighbor, removes incorrectly classified points from training data
    # continues to remove until performance on the testing set decreases

    def EKNN(self, tune=False):

        # initialization

        if tune:
            test = self.tune  # uses tuning sample for when tuning epsilon
        else:
            test = self.samples[9]  # selects the last sample for testing data if not tuning
        self.train = pd.concat(self.samples[0:9])  # initializes training data from samples
        train = pd.DataFrame(self.train)  # initializes local training set to edit
        edited = pd.DataFrame(train) # initializes local edited training set to test performance for reduced data set
        len = self.train.shape[0]  # Creates variable to check if we have gone over every element in the training data
        count = 0  # initializes how many times we have iterated for incremental testing
        increment = (len // 10)  # how often to test performance
        bestSet = pd.DataFrame(train)  # initializes variable to store previous edited set
        bestPerf = self.performance(test,train)  # sets initial performance

        for x in (train.to_dict("index").keys()): # iterates through training data
            count += 1  # amount of elements gone through in training data
            edited = self.edit(x, edited)  # editing returns dataframe with or without x if correct/incorrect
            if edited.shape[0] == self.k:  # prevents from getting to small of a set may not be necessary
                break
            if count % increment == 0:  # if we should incrementally test the current edited set
                currentPerf = self.performance(test, edited)  # gets performance for current edited set
                #  checks if performance decreased
                if (self.classification and currentPerf < bestPerf) or (not self.classification and currentPerf > bestPerf):
                    increased = False
                    break
                else:
                    # updates best performing edited set
                    bestPerf = currentPerf
                    bestSet = pd.DataFrame(edited)


        if self.clusters:  # checks if clusters is already set
            if bestSet.shape[0] < self.clusters:  # only updates number of clusters if less points in training data
                self.clusters = bestSet.shape[0]
        else:
            self.clusters = bestSet.shape[0]  # initializes number of clusters to use
        return bestPerf  # returns the best performance of the algorithm

    # edit
    # ----
    # used with EKNN
    # gets prediction for element being edited
    # returns edited dataframe
    # element removed if incorrect
    def edit(self, x, train, video=False):
        train.drop('Dist', axis=1, inplace=True, errors='ignore')  # removes dist to prevent errors
        idf = train.loc[x, :]  # initializes training point to check to edit
        actual = idf.iloc[-1]  # actual class or regression value
        x = train.loc[x, :].to_dict()  # picks singular element from training data
        train = train.drop(index=idf.name)  # drops that element from training data
        i = np.array(idf.iloc[:-1].to_numpy())  # sets element to array for optimized calculation and removes class data
        trainV = np.array(train.iloc[:,:-1].to_numpy()) # sets training data to array for optimized calculation and removes class data

        if self.discrete:
            t, p = dist.initialize(train)  # initializes VDM for each element in training data
            # uses VDM if discrete variables
            distance = dist.VDM(t, x, p)
        else:  # uses Euclidean in continuous
            distance = np.sum((i - trainV) ** 2, axis=1) ** (1 / 2)  # calculates euclidean distance to all training elements from test element

        train['Dist'] = distance  # adds distance of each element to new data frame column, to determine classes
        neighbors = np.sort(train.to_numpy()[:,-2:], axis=0)[:int(self.k),:]  # finds the k-nearest neighbors
        train.drop('Dist', axis=1, inplace=True, errors='ignore')  # removes distance column
        px = self.predict(neighbors)  # initializes prediction
        # for regression data sets
        if not self.classification:
            error = abs(px - actual)  # calculates actual error
            allowederror = px*self.eps  # calculates allowed error by multiplying the predicted value by epsilon
            if error < allowederror:  # if regression estimate is within epsilon of actual value
                train = pd.concat([pd.DataFrame(idf).transpose(), train])  # re adds element to training data if correct
            elif video:
                print("Actual:",actual,"\nPredicted:",px,"\n edited out" )  # for demonstration
                return
        # for classification data sets
        else:
            if self.correct(px, actual):  # if properly classified added back to training data
                train = pd.concat([pd.DataFrame(idf).transpose(), train])
            elif video:
                print("Actual:",actual,"\nPredicted:",px,"\n edited out" )
                return

        return train  # returns data set with or without the point to be edited

    # performance
    # -------
    # tests performance of Eknn on the testing data
    # similar to kNN
    def performance(self, test, train):
        # initializes testing and training vectors and list to capture performance of each element
        testV = np.array(test.to_numpy())
        trainV = np.array(train.to_numpy())
        tPerf = []
        # goes through all elements of test data
        if self.discrete:
            t, p = dist.initialize(train)  # initializes VDM for each element in training data
            testD = test.to_dict('index')  # puts data frame to dict for iteration
            for x in testD.values():  # iterates
                distance = dist.VDM(t, x, p)  # gets distances
                train['Dist'] = distance   # applys distances to data frame
                neighbors = np.sort(train.to_numpy()[:,-2:], axis=0)[:int(self.k),:]  # gets k nearest
                # train.drop('Dist', axis=1, inplace=True)
                px = self.predict(neighbors)  # sets prediction
                tPerf.append(self.correct(px, list(x.values())[-1]))  # returns squared error / amount correct
            correct = self.evaluate(tPerf) / testV.shape[0]  # returns mean squared error / % correct
        else:
            for x in testV: # iterates through array elements
                distance = np.sum((x[:-1] - trainV[:,:-1])**2, axis=1) ** (1 / 2)  # calculates distance to all training elements from test element
                train['Dist'] = distance
                neighbors = np.sort(train.to_numpy()[:, -2:], axis=0)[:int(self.k), :]
                # neighbors = train.nsmallest(self.k, 'Dist').iloc[:, -2:].values
                px = self.predict(neighbors)
                tPerf.append(self.correct(px, x[-1]))  # returns squared error / amount correct
            correct = self.evaluate(tPerf) / testV.shape[0]  # returns mean squared error / % correct
        train.drop('Dist', axis=1, inplace=True, errors='ignore')
        return correct  # returns performance

    # KNN
    # ---
    # calculates performance of nearest neighbor algorithm based on training data and testing data
    # for regression eps should be 0
    def KNN(self, tune=False, means=False, video=False):
        if tune:  # uses tuning sample if tuning
            test = self.tune
        else:  # uses 10th sample for test if being run normally
            test = self.samples[9]
        self.train = pd.concat(self.samples[0:9])  # intializes global training data
        if means:
            self.train = self.means  # used to use the means from k-means output
        self.train.drop('Dist', axis=1, inplace=True, errors='ignore')  # to prevent errors
        train = pd.DataFrame(self.train)  # initializes local training data

        # initializes arrays of vectors

        trainV = np.array(train.to_numpy())
        testV = np.array(test.to_numpy())
        performance = []

        # checks data type
        if not self.discrete:
            # iterates through each vector
            for i in testV:
                distance = np.sum((i[:-1] - trainV[:,:-1])**2,axis=1)**(1/2)  # calculates euclidean distance to each training point
                train['Dist'] = distance  # appends distance to data frame

                neighbors = np.sort(train.to_numpy()[:, -2:], axis=0)[:int(self.k), :] # finds kth smallest distances
                px = self.predict(neighbors)  # predicts value for i

                if video:  # for demonstration
                    neighbors = np.sort(train.to_numpy()[:, -2:], axis=0)[:int(self.k), :]
                    print(neighbors)
                    print(px)
                    return

                performance.append(self.correct(px, i[-1]))  # adds result to list

        else:
            t, p = dist.initialize(train)  # initializes VDM distances
            testD = test.to_dict('index')  # converts to dictionary because vdm worked better with them ig
            for x in testD.values():  # goes through each value
                distance = dist.VDM(t, x, p)  # gets distance to each training point
                train['Dist'] = distance  # adds distance to df
                neighbors = np.sort(train.to_numpy()[:, -2:], axis=0)[:int(self.k), :]  # finds kth smallest distances
                px = self.predict(neighbors) # predicts value for x
                actual = x['class']  # gets actual value
                if video:
                    neighbors = np.sort(train.to_numpy(), axis=0)[:int(self.k), :]
                    print(neighbors)
                    print(px)
                    return

                performance.append(self.correct(px, actual))  # adds result to list

        correct = self.evaluate(performance) / test.shape[0] # calculates MSE or accuracy

        return correct # returns performance result

    # distance
    # --------
    # not used as simple to implement inline but shows decision for distance calculation
    def distance(self, x, trainV=[], t=[], p=[]):
        if self.discrete:
            distance = dist.VDM(t, x, p)
        else:
            distance = np.sum((x-trainV)**2,axis=1)**(1/2)
        return distance

    # K-Means
    # ---------
    # initializes k random clusters given the amount of clusters set by Edited nearest neighbor
    # Continues finding mean cluster until converges
    def Kmeans(self):
        np.seterr(invalid='ignore')  # ignores div 0 errors
        # initialize random centroids and variable to check if centroids changed
        #self.EKNN()  # sets number of clusters
        self.train = pd.concat(self.samples[0:9])  # initializes training data
        self.train.drop('Dist', axis=1, inplace=True, errors="ignore") # to prevent errors
        train = pd.DataFrame(self.train)  # initializes local training set
        OGcentroids = pd.DataFrame(train.sample(n=self.clusters,replace=False))  # takes k sample for random initial mean centroids
        changed = True  # initiliazes boolean variable to indicate it centroids changed
        C = np.array(OGcentroids.to_numpy()) # intialized Centroid numpy array
        means = np.zeros_like(C) # initializes array for each training point be added to
        old = np.array(OGcentroids.to_numpy())  # initializes array for centroid data to compare each iteration
        Col= OGcentroids.columns.values  # stores the headings
        centroids = pd.DataFrame(old, columns=Col)

        while changed:
            means = np.zeros_like(old)  # updates means to 0
            centroids = pd.DataFrame(old,columns=Col)  # creates dataframe of each cluster
            Ccounts = [0 for i in range(centroids.shape[0])]  # creates empty list of counts for each cluster
            C = np.array(centroids.to_numpy())  # reinitialized mean centroids
            if self.discrete:  # decides distance function to use
                trainD = train.to_dict('index')  # training data to dictionary for iteration
                t, p = dist.initialize(centroids)   # initializes VDM arrays given centroids as training data
                for x in (trainD.values()):  # goes through all data points
                    distances = dist.VDM(t, x, p, means=centroids)  # gets distance from each centroid to data point
                    c = np.argmin(distances[:,-1])  # returns closest centroid to data point
                    xi = np.array((list(x.values())))  # creates array from training point for np calculations
                    means[c] = means[c,:]+xi  # adds training point to centroid values
                    Ccounts[c] += 1  # adds count for centroid for average
                means = pd.DataFrame(means)  # converts means to dataframe
                means['count'] = Ccounts  # adds count for each cluster
                # gets average for each cluster
                means = means[means['count']  > 0].divide(means[means['count']  > 0]['count'],axis=0).round(decimals=0)

                # new clusters to array and removes counts
                m = np.array(means.to_numpy()[:,:-1])

                if np.array_equal(m,old):  # Checks if centroids changed
                    changed = False
                    break
                else:
                    old = np.array(m)  # updates centroids


            else:
                trainV = np.array(train.to_numpy())  # puts training data to numpy
                for i in trainV:  # going through all data points
                    distances = np.sum((i[:-1] - C[:,:-1]) ** 2, axis=1) ** (1 / 2)  # calculates distance to all centroids
                    centroids['Dist'] = distances  # connects distances to dataframe
                    c = np.argmin(np.array(centroids.to_numpy())[:,-1])  # picks centroid with the closest distance
                    means[c] = means[c, :] + i  # adds training point to centroid
                    Ccounts[c] += 1  # adds count for centroid
                means = pd.DataFrame(means)  # converts means to dataframe
                means['count'] = Ccounts  # adds count for each cluster
                # gets average for each cluster
                means = means[means['count'] > 0].divide(means[means['count'] > 0]['count'], axis=0).round(decimals=0)

                # new clusters to array and removes counts
                m = np.array(means.to_numpy()[:, :-1])
                if np.array_equal(m,old):  # Checks if centroid changed
                    centroids = pd.DataFrame(old)
                    changed = False
                    break

                else:
                    old = np.array(m)


        # classify dependent variable of each centroid using KNN with training data
        if not self.discrete:
            testV = m  # initializes clusters to array
            index = 0  # initializes index pointer
            for i in testV:  # goes through each cluster
                distance = np.sum((i[:-1] - trainV[:,:-1])**2,axis=1)**(1/2)  # gets distance from cluster to each training point
                train['Dist'] = distance  # sets distances
                neighbors = np.sort(train.to_numpy()[:, -2:], axis=0)[:int(self.k), :]  # finds neighbors
                px = self.predict(neighbors)  # gets prediction for cluster
                centroids.iloc[index,-1] = px  # assigns prediction to cluster dependent variable
                index += 1

        else:
            t, p = dist.initialize(train)
            testD = centroids.to_dict('index')
            index = 0
            for x in testD.values():
                distance = dist.VDM(t, x, p)  # gets VDM distance
                train['Dist'] = distance
                neighbors = np.sort(train.to_numpy()[:, -2:], axis=0)[:int(self.k), :]
                px = self.predict(neighbors)
                centroids.iloc[index, :]['class'] = px # updates class for x to predicted
                index += 1
        self.means = centroids
        return self.KNN(means=True)

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
            return abs(prediction-actual)**2  # squared error

    # evaluate
    # --------
    # returns accumulated performance based on output type
    def evaluate(self, performance):
        if self.classification:
            return performance.count(True)
        else:
            return sum(performance)

    # classify
    # --------
    # gets class with most votes
    # performs plurality vote
    def classify(self, neighbors):

        neighbors = pd.DataFrame(neighbors)
        weights = pd.DataFrame(neighbors.groupby(by=[0])[1].sum())
        Px = weights[weights[1] == weights[1].min()].index.values[0]

        return Px

    # regression
    # ----------
    # gets predicted output using gaussian kernel
    # uses kernel smoothing function
    def regression(self, kN):

        numer = sum([self.gaussianK(i[1])*i[0] for i in kN])
        denom = sum([self.gaussianK(i[1]) for i in kN])
        px = (numer / denom)
        return px

    def gaussianK(self, u):
        x = (u**2) / (2*self.bandwith**2)
        y = exp(-x)
        z = y / (2*np.pi)**(1/2)
        return z

 # tuning
 # ---------------------------------------------------------------
    # tuneit
    # ----
    # tunes all hyperparameters for the data set
    # if classification only tunes K
    # if regression then performs grid search of bandwidth and k and then epsilon
    # results from past tests can be saved
    def tuneit(self,test=False):
        if self.name == 'abalone.data':
            self.k = 68
            self.bandwith = 10
            self.eps = 0.01
            return "K, Bandwith: ",self.k,self.bandwith," epsilon: " + str(self.eps)
        #elif self.name == 'machine.data':
        #    self.k = 27
        #    self.bandwith = 4852.95
        #    self.eps = 0.01
        #    return "K, Bandwith: ",self.k,self.bandwith," epsilon: " + str(self.eps)
        elif self.name == 'forestfires.data':
            self.k = 1
            self.bandwith = 75
            self.eps = 0.01
            return "K, Bandwith: ",self.k,self.bandwith," epsilon: " + str(self.eps)
        self.k = 1
        if self.classification:
            return "K, performance: "+str(self.tuneK())  # tunes only k
        else:  # tunes k, bandwidth, and epsilon
            return "K, Bandwith: "+str(self.tuneBandwidth(test=test)), " MSE: "+str(self.tuneEpsilon(test=test))+" epsilon: "+str(self.eps)

    # tuneEpsilon
    # -------------
    # used for regression
    # tunes epsilon to minimize MSE of the data set
    def tuneEpsilon(self,):

        # pick range of epsilon
        it = np.arange(.01, .2, .01)
        perf = {}  # dictionary of Eps value : performance
        for i in (it):  # goes through range
            self.eps = i  # sets epsilon
            mse = 0  # init mean sq error
            for t in range(2):  # how many tests to avg
                mse += self.EKNN(tune=True)
            perf[i] = mse/2

        low = min(perf, key=perf.get)  # gets epsilon of lowest mse
        self.eps = low  # sets eps to best performing one
        lists = sorted(perf.items())  # for grph
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
    #@profiler
    def tuneBandwidth(self,test=False):
        # preset values from prev tests
        if self.name == 'machine.data':
            x = 100000 # 10000
            y = 10000  # 100
            self.bandwith = 1
        elif self.name == 'forestfires.data':
            x = 1000 # 100
            y = 3000 # 30
            self.bandwith = 1
        else:
            x = 100 # 10
            y = 100 # 1
            self.bandwith = 1

        results = []
        self.k = int(round(self.train.shape[0]**(1/2)))  # tries to find where to start search for k
        sqrk = self.KNN(tune=True)
        self.k = 1
        k = self.KNN(tune=True)
        max = round(self.train.shape[0]**(1/2)+10)
        step = max // 10
        if self.classification:
            if k > sqrk:
                max = 5
                step = 1
        else:
            if k < sqrk:
                max = 5
                step = 1
        if test:
            it = np.arange(1,3)
            times = 2
        else:
            it = np.arange(1,50,2)
            times = 25
        # most of this above is random testing stuff to optimize runtimes depending on bug testing or actual tuning

        for k in it:  # goes through k values
            self.k = int(k)  # sets k
            for h in np.random.randint(y, 10*x, times)/100:  # picks x random values for bandwidth in the predefined range
                self.bandwith = h  # sets bandwidth
                MSE = self.KNN(tune=True)  # gets result
                results.append([k,h,MSE])  # appends result

        low = min(results, key=lambda x: x[2])  # gets k and bandwidth of lowest error

        # for heat map graph
        results = np.array(results)
        plt.scatter(results[:, 0], results[:, 1], c=results[:, 2], cmap='hot')
        plt.xlabel('k_neighbors', fontsize=14)
        plt.ylabel('kernel_width', fontsize=14)
        plt.colorbar().set_label('Mean Squared Error')
        plt.savefig("tuning/" + "kandbandwith."+self.name[:-4]+'png')
        plt.clf()

        k,h = low[:-1]  # sets to best performing values

        self.k = k  # sets k
        self.bandwith = h  # sets bandwidth
        return k, h

    # tuneK
    # --------------------------------------
    # tunes k for the data set to use with KNN EKNN and Kmeans
    # sets the k for the data set, carries through all functions
    # only classification data sets
    def tuneK(self):
        tune = {}
        self.train = pd.concat(self.samples)
        for k in range(1,50,5):  # range to try
            self.k = k  # setting k
            performance = self.KNN(tune=True)  # getting perf
            tune[k] = performance  # adding result

        kk = max(tune, key=tune.get)  # gets best performing k value

        self.k = kk  # sets k to best one

        # graph out

        lists = sorted(tune.items())
        x, y = zip(*lists)
        plt.plot(x, y)
        plt.xlabel("K neighbors")
        if self.classification: plt.ylabel("Prob")
        else: plt.ylabel("MSE")
        plt.savefig("tuning/ktune"+self.name[:-4]+'png')
        plt.clf()
        return kk, tune[kk]


#   i know there are lots of repeated code, and it could be better structured, but it works after many bugs and im proud
#   i have probably spent over 24hrs in the past week working on this and pulled an all nighter last night :/
#   thanks for reading

