import pandas as pd
import numpy as np
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df = pd.read_csv("/home/humza/Downloads/train.csv")
types = df.dtypes
stringArray = []
for key in types.keys():
    if types[key]== "object":
        stringArray.append(key)
print(stringArray)
for item in stringArray:
    dummyList = df[item].tolist()
    df.drop(item, axis=1)
    le.fit(dummyList)
    dummyList = le.transform(dummyList)
    df[item] = dummyList

output = df["Survived"].to_numpy(dtype = np.float64)
df.drop("Survived", axis = 1, inplace = True)
print(df.columns)
input = df.to_numpy(dtype = np.float64)
print(input.shape)
print(output.shape)



