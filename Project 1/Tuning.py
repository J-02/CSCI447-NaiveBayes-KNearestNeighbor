import matplotlib.pyplot as plt
import numpy as np
import Naivetest as Nt
import os
from tqdm import tqdm


# iterates through bin sizes
# only glass and iris need tuning all others are not binned


for file in os.listdir("Data/unbinned/"):
    p = []
    bins = []

    for l in tqdm(range(50), desc= file+" tuning..."):
        current = Nt.NaiveBayes("Data/unbinned/"+file)
        b = l*2
        current.bin(b)
        avg = []
        for i in range(20):
            trainData = current.df.groupby('class', group_keys=False).apply(lambda x: x.sample(frac=.5))
            testData = current.df.drop(trainData.index)
            trainP = current.train(trainData)
            avg.append(current.test(testData, trainP)[1])

        bins.append(b)
        p.append(np.average(avg))

    plt.scatter(bins,p, label=file[0:-5])
    plt.legend()


plt.savefig('Results/tuning'+'.png')
plt.show()