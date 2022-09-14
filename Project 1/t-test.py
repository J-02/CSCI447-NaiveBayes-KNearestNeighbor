import numpy as np

import CrossValidation as CV

results = CV.main()

control = results[0]
scrambled = results[1]
files = results[2]
difference = np.subtract(control,scrambled)
mean = np.mean(difference)
stdev = np.std(difference)

t = mean/(stdev/(np.sqrt(len(difference))))

print(t)
