import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import *

from MLBook.kNN import kNN

datingDataMat, datingLabels = kNN.file2matrix("datingTestSet2.txt")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # type:Axes3D
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], datingDataMat[:, 0], 'z',
           10 * array(datingLabels), 150 * array(datingLabels))
plt.show()
