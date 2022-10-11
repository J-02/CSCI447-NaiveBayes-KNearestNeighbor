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
            self.eps = 0.1
            self.bandwith = 1
            self.result = "MSE"
        self.clusters = 0

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
    # todo: if performance degrades after removing a point, readd it see if works better
    # todo: doesn't get to check if all points are noise
    def EKNN(self, tune=False):
        if tune:
            test = self.tune
        else:
            test = self.samples[9]
        self.train = pd.concat(self.samples[0:9])
        # initialization
        train = pd.DataFrame(self.train)  # initializes local training set to edit
        edited = pd.DataFrame(train) # initializes local edited training set to test performance for reduced data set
        len = self.train.shape[0]  # Creates variable to check if we have gone over every element in the training data
        if self.classification: bestPerf = 0  # determines what value performance starts at
        if not self.classification: bestPerf = np.infty  # lower is better for regression performance value MSE
        count = 0  # initializes how many times we have edited
        increment = (len // 10)  # how often to test performance
        prevSet = pd.DataFrame(train)  # initializes variable to store previous edited set
        bestPerf = self.performance(test,train)  # sets initial performance
        # for regression data sets
        if not self.classification:
            for x in (train.to_dict('index').keys()):  # iterating through each element in training data

                count += 1  # adds to amount of elements gone through in training data

                edited = self.edit(x, prevSet)

                if edited.shape[0] < prevSet.shape[0]:
                    currentPerf = self.performance(test, edited)
                    if currentPerf > bestPerf:
                        edited = pd.concat([pd.DataFrame(train.loc[x, :]).transpose(), edited])
                    else:
                        bestPerf = currentPerf
                    prevSet = pd.DataFrame(edited)
                #prevSize = prevSet.shape[0]
                #currentSize = edited.shape[0]
                #if count % increment == 0 :  # tests if a point has been edited out
                    # tests if error increased
                    #correctPerf = self.performance(test, edited)
                    #if correctPerf > prevPerf:
                        #self.clusters = prevSet.shape[0]
                        # returns best performance
                        #return prevPerf
                    #else:
                    # updates performance if it improved and saves the edited training data
                    # prevPerf = correctPerf
                # prevSet = pd.DataFrame(edited)


        # for classification
        else:

            for x in (train.to_dict('index').keys()):
                count += 1
                edited = self.edit(x, edited)  # returns data set with or without x depending if correct

                if edited.shape[0] < prevSet.shape[0]:
                    currentPerf = self.performance(test, edited)
                    if currentPerf < bestPerf:
                        edited = pd.concat([pd.DataFrame(train.loc[x, :]).transpose(), edited])
                    else:
                        bestPerf = currentPerf
                    prevSet = pd.DataFrame(edited)

                    
                #if edited.shape[0] < prevSet.shape[0]:  # tests a point has been edited out
                #    correct = self.performance(test, edited)
                    # tests if error increased
                #    if correct < prevPerf:
                #        self.clusters = prevSet.shape[0]
                        # returns best performance
                #        return prevPerf
                    #else:
                        # updates performance if it improved and saves the edited training data
                        #prevPerf = correct
                        #prevSet = edited
                        #performance.append(correct)
        self.clusters = prevSet.shape[0]
        return bestPerf

    # edit
    # ----
    # gets prediction for element being edited
    # returns edited dataframe
    # element removed if incorrect
    def edit(self, x, train):

        idf = train.loc[x, :]
        actual = idf.iloc[-1]
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
            error = abs(px - actual)  # calculates actual error
            allowederror = self.eps*actual  # calculates allowed error by multiplying the actual value by epsilon
            if error < allowederror:  # if regression estimate is within epsilon of actual value
                train = pd.concat([pd.DataFrame(idf).transpose(), train])  # readds element to training data if correct
        # for classification data sets
        else:
            if self.correct(px, actual):  # if properly classified added back to training data
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
                distance = np.sum((x[:-1] - trainV[:,:-1]) ** 2, axis=1) ** (1 / 2)  # calculates distance to all training elements from test element
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
    def KNN(self, tune=False, means=False):
        if tune:  # uses tuning sample if tuning
            test = self.tune
        elif means:  # uses cluster means to evaluate k means performance
            test = self.means
        else:  # uses 10th sample for test if being run normally
            test = self.samples[9]
        self.train = pd.concat(self.samples[0:9])
        self.train.drop('Dist', axis=1, inplace=True, errors='ignore')
        train = pd.DataFrame(self.train)  # initializes local training data

        # initializes arrays of vectors

        trainV = np.array(train.to_numpy())
        testV = np.array(test.to_numpy())
        performance = []

        # checks data type
        if not self.discrete:
            # iterates through each vector
            for i in testV:
                if means:
                    distance = np.sum((i[:-1] - trainV[:, :-1]) ** 2, axis=1) ** (1 / 2)
                distance = np.sum((i[:-1] - trainV[:,:-1])**2,axis=1)**(1/2)  # calculates euclidean distance to each training point
                train['Dist'] = distance  # appends distance to data frame
                neighbors = train.nsmallest(self.k, 'Dist').iloc[:,-2:].values  # finds kth smallest distances
                train.drop('Dist', axis=1, inplace=True)  # removes distance values
                px = self.predict(neighbors)  # predicts value for i
                performance.append(self.correct(px, i[-1]))  # adds result to list

        else:
            t, p = dist.initialize(train)  # initializes VDM distances
            testD = test.to_dict('index')  # converts to dictionary because vdm worked better with them ig
            for x in testD.values():  # goes through each value
                distance = dist.VDM(t, x, p)  # gets distance to each training point
                train['Dist'] = distance  # adds distance to df
                neighbors = train.nsmallest(5, 'Dist').iloc[:,-2:].values # finds kth smallest distances
                train.drop('Dist', axis=1, inplace=True) # removes distance values
                px = self.predict(neighbors) # predicts value for x
                performance.append(self.correct(px, list(x.values())[-1])) # adds result to list

        correct = self.evaluate(performance) / testV.shape[0]

        return correct

    def distance(self, x, trainV=[], t=[], p=[]):
        if self.discrete:
            distance = dist.VDM(t, x, p)
        else:
            distance = np.sum((x-trainV)**2,axis=1)**(1/2)
        return distance


    def Kmeans(self):
        np.seterr(invalid='ignore')
        # initialize random centroids and variable to check if centroids changed
        self.EKNN()  # sets number of clusters
        self.train = pd.concat(self.samples[0:9])
        self.train.drop('Dist', axis=1, inplace=True, errors="ignore")
        train = pd.DataFrame(self.train)  # initializes local training set
        OGcentroids = pd.DataFrame(train.sample(n=self.clusters,replace=False))  # takes k sample for random initial mean centroids
        changed = True  # initiliazes boolean variable to indicate it centroids changed
        C = np.array(OGcentroids.to_numpy())
        means = np.zeros_like(C)# initializes array for centroid data to change
        old = np.array(OGcentroids.to_numpy())  # initializes array for centroid data to compare each iteration
        Col= OGcentroids.columns.values  # stores the headings
        lastCcounts = []  # counts to each centroid

        # continue until centroids converge
        #pbar = tqdm()
        n = 0
        while changed:
            means = np.zeros_like(old)
            centroids = pd.DataFrame(old,columns=Col)  # creates dataframe of each cluster
            Ccounts = [0 for i in range(centroids.shape[0])]  # creates empty list of counts for each cluster

            if self.discrete:  # decides distance function to use
                trainD = train.to_dict('index')  # training data to dictionary for iteration
                t, p = dist.initialize(centroids)   # initializes VDM arrays given centroids as training data
                for x in (trainD.values()):  # goes through all data points
                    distances = dist.VDM(t, x, p, means=centroids)  # gets distance from each centroid to data point
                    c = np.argmin(distances[:,-1])  # returns closest centroid to data point
                    xi = np.array((list(x.values())))  # creates array from training point for np calculations
                    means[c] = means[c,:]+xi  # adds training point to centroid values
                    Ccounts[c] += 1  # adds count for centroid for average

                zeros = []
                for i in Ccounts:

                    if i == 0:
                        zeros.append(i)
                Acounts = np.delete(Ccounts, zeros, 0)
                Ameans = np.delete(means, zeros, 0)
                AC = np.delete(C, zeros, 0)
                C = AC

                m = np.rint(means / np.array(Ccounts)[:,None])  # Calculates mean of centroid rounded to the nearest int
                if np.array_equal(m,old):  # Checks if centroids changed
                    changed = False
                    break
                # if Ccounts == lastCcounts:
                #    changed = False
                #    break
                else:
                    old = np.array(m)  # updates centroids
                    lastCcounts = list(Ccounts)

            else:
                trainV = np.array(train.to_numpy())  # puts training data to numpy
                for i in trainV:  # going through all data points
                    distances = np.sum((i[:-1] - C[:,:-1]) ** 2, axis=1) ** (1 / 2)  # calculates distance to all centroids
                    centroids['Dist'] = distances  # connects distances to dataframe
                    c = np.argmin(np.array(centroids.to_numpy())[:,-1])  # picks centroid with the closest distance
                    centroids.drop('Dist', axis=1, inplace=True)  # removes distance column from dataframe
                    means[c] = means[c, :] + i  # adds training point to centroid
                    Ccounts[c] += 1  # adds count for centroid
                zeros = []
                for i in Ccounts:

                    if i == 0:
                        zeros.append(i)
                Acounts = np.delete(Ccounts,zeros,0)
                Ameans = np.delete(means,zeros,0)
                AC = np.delete(C,zeros,0)
                C = AC
                m = Ameans / Acounts[:, None] # calculates mean centroid not rounded
                if np.array_equal(m,old):  # Checks if centroid changed
                    centroids = pd.DataFrame(old)
                    changed = False

                    break
                #if Ccounts == lastCcounts:
                #    changed = False
                #    break
                else:
                    old = np.array(m)
                    lastCcounts = list(Ccounts)
            n += 1
            #pbar.update(n)

        # classify dependent variable of each centroid using KNN with training data
        if not self.discrete:
            testV = m  # initializes clusters to array
            index = 0
            for i in testV:  # goes through each cluster
                distance = np.sum((i[:-1] - trainV[:,:-1])**2,axis=1)**(1/2)  # gets distance from cluster to each training point
                train['Dist'] = distance  # sets distances
                neighbors = train.nsmallest(self.k, 'Dist').iloc[:,-2:].values  # finds neighbors
                train.drop('Dist', axis=1, inplace=True)
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
                neighbors = train.nsmallest(5, 'Dist').iloc[:,-2:].values
                train.drop('Dist', axis=1, inplace=True)
                px = self.predict(neighbors)
                centroids.iloc[index,'class'] = px # updates class for x to predicted
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
            return abs(prediction-actual)**2

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
        #if denom == 0:
        #   denom = 1

        px = (numer / denom)
        #print('Prediction:', px)

        return px

    def gaussianK(self, u):
        x = (u**2) / (2*self.bandwith**2)
        x = exp(-x)
        x = x / (2*np.pi)**(1/2)
        return x

 # tuning
 # ---------------------------------------------------------------
    # tuneit
    # ----
    # tunes all hyperparameters for the data set
    # if classification only tunes K
    # if regression then performs grid search of bandwidth and k and then epsilon
    @timeit
    def tuneit(self,test=False):
        if self.classification:
            return "K, performance: "+str(self.tuneK())
        else:
            return "K, Bandwith: "+str(self.tuneBandwidth(test=test)), "epsilon: "+str(self.eps)+" MSE: "+str(self.tuneEpsilon(test=test))

    # tuneEpsilon
    # -------------
    # used for regression
    # tunes epsilon to minimize MSE of the data set
    def tuneEpsilon(self,test=False):
        if test:
            it = np.arange(.1,.3,.1)
        else:
            it = np.arange(.01,1,.1)
        if self.name == "abalone.data":
            it = np.arange(.01, .5, .01)
        perf = {}
        for i in it:
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

    def tuneBandwidth(self,test=False):
        if self.name == 'machine.data':
            x = 500000 # 5000
            y = 10000  # 100
            self.bandwith = 100
        elif self.name == 'forestfires.data':
            x = 10000 # 100
            y = 10000 # 10
            self.bandwith = 10
        else:
            x = 10 # 0.1
            y = 10 # 0.01
            self.bandwith = 0.1
        results = []
        self.k = round(self.train.shape[0]**(1/2))
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
            it = np.arange(1,max+5,2)
            times = 10
        for k in tqdm(it):
            self.k = k
            for h in np.random.randint(y, 10*x, times)/1000:
                self.bandwith = h
                MSE = self.KNN(tune=True)
                results.append([k,h,MSE])

        low = min(results, key=lambda x: x[2])

        # For easier access of columns, convert to numpy array
        results = np.array(results)
        # Now we visualize.
        plt.scatter(results[:, 0], results[:, 1], c=results[:, 2], cmap='hot')
        plt.xlabel('n_neighbors', fontsize=14)
        plt.ylabel('kernel_width', fontsize=14)
        plt.colorbar().set_label('Mean Squared Error')
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
        self.train = pd.concat(self.samples)
        for k in range(1,round(self.train.shape[0]**(1/2)+5)):
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
        plt.savefig("tuning/ktune"+self.name[:-4]+'png')
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

