import sklearn
import numpy as np
import pandas

df = pandas.read_csv("train.csv")

labels = df.iloc[:, 0:2]  # label
data = df.drop('Survived', 1)  # train data
print(labels)
print("=========================================")
print(data)
