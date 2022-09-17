
import pandas as pd
import numpy as np
import Naivetest as nt
import os


# ------------------------------------------------------------------------
# bin
# This creates data files in the Data directory that are discretized or binned to the given bin size
def bin():
    for file in os.listdir("Data/unbinned/"):
        print(file)
        current = nt.NaiveBayes('Data/unbinned/'+file)
        #print(current.df)
        if file.__contains__('glass'):
            current.bin(20) #change this number to alter glass.data bin size
        if file.__contains__('iris'):
            current.bin(6) #change this number to alter iris.data bin size
        #print(current.df)
        current.df.to_csv("Data/"+file)

