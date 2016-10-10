import numpy as np
a = np.asarray([[1,2],[2,3]])
b = a**2
print b
print np.concatenate((a,b), axis = 1)


