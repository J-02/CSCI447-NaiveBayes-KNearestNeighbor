
import pandas as pd
import numpy as np

file = pd.read_csv('Data/breast-cancer-wisconsin.data',index_col=0)

print(file)

#file = file.replace("y", 1)
#file = file.replace('n', 0)
#file = file.replace('?', 2)
#print(file)
file.to_csv('Data/breast-cancer-wisconsin.data')
