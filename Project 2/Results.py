import os

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

import NNV as nn


@nn.timeit
def results():
    runs = 1
    for file in (file for file in os.listdir("Data") if file.endswith('.data')):
            test1 = nn.NearestNeighbor(file)
            print("Tuning ", file)
            print("Using:", test1.tuneit())

            if test1.k == 1:
                minR = 1
                maxR = 6

            else:
                minR = max(1, test1.k - 2)
                maxR = minR + 5

            KNNResults = {}
            print("KNN 10 fold CV")
            for i in (np.arange(minR, maxR)):
                test1.k = i
                results = 0
                for l in range(runs):
                    results += test1.KNN()
                    test1.samples.append(test1.samples.pop(0))
                    test1.train = pd.concat(test1.samples[0:9])

                results = results / runs
                print("K = {} \nResult = ".format(test1.k), results)

                KNNResults[i] = results

            lists = sorted(KNNResults.items())
            x, y = zip(*lists)
            plt.plot(x, y)
            plt.title("K Nearest Neighbor")
            plt.xlabel("K neighbors")
            if test1.classification:
                plt.ylabel("Accuracy")
            else:
                plt.ylabel("MSE")
            plt.savefig("results/KnnResults" + test1.name[:-4] + 'png')
            plt.clf()
            print(pd.DataFrame([KNNResults]).to_latex())

            EKNNResults = {}
            print("EKNN 10 fold CV")
            for i in (np.arange(minR, maxR)):
                test1.k = i
                results = 0
                for l in range(runs):
                    results += test1.EKNN()
                    test1.samples.append(test1.samples.pop(0))
                    test1.train = pd.concat(test1.samples[0:9])

                results = results / runs
                print("K = {} \nResult = ".format(test1.k), results)

                EKNNResults[i] = results

            lists = sorted(EKNNResults.items())
            x, y = zip(*lists)
            plt.plot(x, y)
            plt.title("Edited K Nearest Neighbor")
            plt.xlabel("K neighbors")
            if test1.classification:
                plt.ylabel("Accuracy")
            else:
                plt.ylabel("MSE")
            plt.savefig("results/EKnnResults" + test1.name[:-4] + 'png')
            plt.clf()
            print(str(pd.DataFrame([EKNNResults]).to_latex()))

            KMeansResults = {}
            print("K-Means 10 fold CV")
            for i in (np.arange(minR, maxR)):
                test1.k = i
                results = 0
                for l in range(runs):
                    results += test1.Kmeans()
                    test1.samples.append(test1.samples.pop(0))
                    test1.train = pd.concat(test1.samples[0:9])

                result = results / runs
                print("K = {} \nResult = ".format(test1.k), result)

                KMeansResults[i] = results

            lists = sorted(KMeansResults.items())
            x, y = zip(*lists)
            plt.plot(x, y)
            plt.title("K-Means")
            plt.xlabel("K neighbors")
            if test1.classification:
                plt.ylabel("Accuracy")
            else:
                plt.ylabel("MSE")
            plt.savefig("results/KMeansResults" + test1.name[:-4] + 'png')
            plt.clf()
            print(pd.DataFrame([EKNNResults]).to_latex())

            final = [KNNResults,EKNNResults,KMeansResults]

            fResults = pd.DataFrame(final)
            print(fResults.to_latex())

def video():
    # data being split
    test1 = nn.NearestNeighbor("forestfires.data")
    test2 = nn.NearestNeighbor("breast-cancer-wisconsin.data")
    test1.tuneit()
    test2.tuneit()
    train = pd.DataFrame(test1.train)
    print("Samples")
    for i in range(len(test1.samples)):
        print("Sample: ",i+1)
        print(test1.samples[i])

    x = train.sample(n=1).to_numpy()
    actual = x[-1,-1]
    x = x[:,:-1]
    y = train.sample(n=1).to_numpy()[:,:-1]
    print(x)
    print(y)

    # distance function
    s1 = (x - y) ** 2
    print(s1)
    s2 = np.sum(s1,axis=1)
    print(s2)
    s3 = s2 ** (1/2)
    print(s3)

    trainV = train.to_numpy()[:,:-1]

    distance = np.sum((x - trainV) ** 2, axis=1) ** (1 / 2)
    print(distance)

    train['distance'] = distance
    neighbors = np.sort(train.to_numpy()[:, -2:], axis=0)[:test1.k, :]
    print(neighbors)
    # kernel function
    numer = sum([test1.gaussianK(i[1]) * i[0] for i in neighbors])
    denom = sum([test1.gaussianK(i[1]) for i in neighbors])
    prediction = (numer / denom)
    print("Prediction: ",prediction,"Actual: ", actual)

    # Neighbors: classification and regression
    test2.KNN(video=True)  # classification
    test1.KNN(video=True)  # regression

    # editing out points
    x = test2.train.sample(n=10,replace=False)
    for i in x.to_dict('index').keys():
        test2.edit(i, test2.train, video=True)

    # regression results
    print(test1.name)
    cv(test1)

    # classification results
    print(test2.name)
    cv(test2)

def cv(test1):
    runs = 10
    print("KNN 10 fold CV")
    results = 0
    for l in range(runs):
        results += test1.KNN()
        test1.samples.append(test1.samples.pop(0))
        test1.train = pd.concat(test1.samples[0:9])

    result = results / runs
    print(test1.result,":",result)

    results = 0


    print("EKNN 10 fold CV")

    for l in range(runs):
        results += test1.EKNN()
        test1.samples.append(test1.samples.pop(0))
        test1.train = pd.concat(test1.samples[0:9])

    result = results / runs
    print(test1.result, ":", result)

    print("K-Means 10 fold CV")
    for l in range(runs):
        results += test1.Kmeans()
        test1.samples.append(test1.samples.pop(0))
        test1.train = pd.concat(test1.samples[0:9])

    result = results / runs
    print(test1.result, ":", result)



video()
