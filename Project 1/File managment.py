
import pandas as pd
import numpy as np
import Naivetest as nt
import os

#file = pd.read_csv('Data/breast-cancer-wisconsin.data',index_col=0)

#print(file)

#file = file.replace("y", 1)
#file = file.replace('n', 0)
#file = file.replace('?', 2)
#print(file)
#file.to_csv('Data/breast-cancer-wisconsin.data')

for file in os.listdir("Data/unbinned/"):
    print(file)
    current = nt.NaiveBayes('Data/unbinned/'+file)
    print(current.df)
    if file.__contains__('glass'):
        current.bin(30)
    if file.__contains__('iris'):
        current.bin(12)
    print(current.df)
    current.df.to_csv("Data/"+file)
