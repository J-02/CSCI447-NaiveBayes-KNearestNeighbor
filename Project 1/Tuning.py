import matplotlib.pyplot as plt
import numpy as np
import Naivetest as Nt
import os
from tqdm import tqdm

#---------------------------------------------------------------------
# iterates through bin sizes
# outputs precision results for varying bin sizes to find one for the best performance
# only glass and iris need tuning all others are not binned

for file in os.listdir("Data/unbinned/"):
    p = []
    r = []
    bins = []

    for l in tqdm(range(31), desc= file+" tuning..."): # max bin size, tqdm is used to show progress and is not required
        current = Nt.NaiveBayes("Data/unbinned/"+file)
        b = l*2
        current.bin(b)
        avgP = []
        avgR = []
        for i in range(5): # sample size
            trainData = current.df.groupby('class', group_keys=False).apply(lambda x: x.sample(frac=.5))
            testData = current.df.drop(trainData.index)
            trainP = current.train(trainData)
            avgR.append(current.test(testData, trainP)[0])
            avgP.append(current.test(testData, trainP)[1])

        bins.append(b)
        p.append(np.average(avgP))
        r.append((np.average(avgR)))

    plt.plot(bins,p, label=file[0:-5]+" Precision")
    plt.legend()


plt.savefig('Results/tuning'+'.png')
plt.show()