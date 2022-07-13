import pandas as pd



import numpy as np

import os

import pyreadstat



from sklearn.preprocessing import OrdinalEncoder




pd.options.display.max_columns=None



# print(os.get)

diary_final = pd.read_csv('data/diary_final.csv')

#diary_final.head(2)



# retrieve numpy array

dataset = diary_final.values




# split into input (X) and output (y) variables

X = dataset[:, :-1]

y = dataset[:,-1]




...

# format all fields as string

X = X.astype(str)



from sklearn.model_selection import train_test_split



# split into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)



oe = OrdinalEncoder()

oe.fit(X_train)