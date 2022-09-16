
import pandas as pd
import numpy as np
import Naivetest as nt
import os

file = pd.read_csv('Data/house-votes-84.data')

print(file)

print(file)
file.to_csv('Data/house-votes-84.data')

def bin():
    for file in os.listdir("Data/unbinned/"):
        print(file)
        current = nt.NaiveBayes('Data/unbinned/'+file)
        #print(current.df)
        if file.__contains__('glass'):
            current.bin(20)
        if file.__contains__('iris'):
            current.bin(6)
        #print(current.df)
        current.df.to_csv("Data/"+file)
