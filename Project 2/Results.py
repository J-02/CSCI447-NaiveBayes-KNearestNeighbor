import NNV as nn
from tqdm import trange
import os


for file in os.listdir("Data"):
    if file.endswith('.data'):
        print(file)
        test1 = nn.NearestNeighbor(file)
        print(test1.KNN())
        print("Using:",test1.tuneit())
        results = 0
        for i in trange(10):
            results += test1.KNN()
            test1.samples.append(test1.samples.pop(0))
        print("Knn:",results/10)
        print("K =",test1.k)

        #print(test1.Kmeans())
        #print(test1.tuneit())
        #print(test1.KNN())
        #print(test1.EKNN())
        #print(test1.result,": ", test1.KNN())