import pandas as pd

import NNV as nn
from tqdm import trange, tqdm
import numpy as np
import os
import matplotlib.pylab as plt
from line_profiler_decorator import profiler

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
    test2.tunit()

    for i in test1.samples:
        print(i)
    x = test1.train.sample(n=1).to_numpy()
    y = test1.train.sample(n=1).to_numpy()
    print(x)
    print(y)

    # distance function
    distance = np.sum((x - y) ** 2, axis=1) ** (1 / 2)
    print(distance)

    # kernal function
    print(test1.gaussianK(distance))

    # Nieghbors: classification and regression
    test2 = nn.NearestNeighbor("breast-cancer-wisconsin.data")
    test2.KNN(video=True)  # classification
    test1.KNN(video=True)  # regression

    # editing out points
    x = test2.train.sample(n=20,replace=False)
    for i in x.to_dict().keys():
        test2.edit(i, test2.train, video=True)

    # regression results
    cv(test1)

    # classification results
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



#video()
results()