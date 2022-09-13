import matplotlib.pyplot as plt
import numpy as np
import Naivetest as Nt
from tqdm import tqdm


# iterates through bin sizes
# only glass and iris need tuning all others are not binned


files = ['Data/glass.data','Data/iris.data']
for file in files:
    p = []
    bins = []
    print(file)
    for l in tqdm(range(15), desc="Loading..."):
        current = Nt.NaiveBayes(file)
        b = l*2
        current.bin(b)
        avg = []
        for i in range(6):
            trainData = current.df.groupby('class', group_keys=False).apply(lambda x: x.sample(frac=.5))
            testData = current.df.drop(trainData.index)
            trainP = current.train(trainData)
            avg.append(current.test(testData, trainP)[1])

        bins.append(b)
        p.append(np.average(avg))

    plt.scatter(bins,p)
    plt.show()