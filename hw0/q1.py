import numpy as np
import sys
data = np.loadtxt(str(sys.argv[2]))
out = open('ans1.txt','w')
sorted_data = np.sort(data[:,int(sys.argv[1])])
length = sorted_data.size
for i in range(length):
    out.write(str(sorted_data[i]))
    if i != length-1:
        out.write(',')
out.close()
