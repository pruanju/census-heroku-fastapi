import pandas as pd
import sys
import pickle
from sklearn.model_selection import train_test_split
import numpy as np


sys.path.append('./starter/ml')

dataframe = pd.read_csv("./data/census.csv")
model = pickle.load(open('./model/Census_model.pickle', 'rb'))
lb = pickle.load(open('./model/Label_model.pickle', 'rb'))
encoder = pickle.load(open('./model/Encoder_model.pickle', 'rb'))

cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
traindata, testdata = train_test_split(dataframe, test_size=0.20)
y = testdata['salary']
X = testdata.drop(['salary'], axis=1)
X_categorical = testdata[cat_features].values
X_continuous = X.drop(*[cat_features], axis=1)
X_cat = encoder.transform(X_categorical)
print(X_cat)
y = lb.transform(y.values).ravel()
X = np.concatenate([X_continuous, X_cat], axis=1)

