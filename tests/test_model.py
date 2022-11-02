import pytest
import pickle
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb


# load the model, encoder and label from disk
model = pickle.load(open('../model/Census_model.pickle', 'rb'))
encoder = pickle.load(open('../model/Encoder_model.pickle', 'rb'))
lb = pickle.load(open('../model/Label_model.pickle', 'rb'))

sys.path.append('../starter/ml')
from data import process_data
from model import *

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

# Load and split the data
df = pd.read_csv("../data/census.csv")


@pytest.fixture
def model():
    mod = pickle.load(open('../model/Census_model.pickle', 'rb'))
    return mod


@pytest.fixture
def data():
    traindata, testdata = train_test_split(df, test_size=0.20)
    X_train, y_train, _, _ = process_data(traindata, categorical_features=cat_features, label="salary", training=True)
    X_test, y_test, _, _ = process_data(testdata, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)
    return ([X_train, y_train, X_test, y_test])


def test_train_model(data):
    X_train = data[0]
    y_train = data[1]
    model_test = train_model(X_train, y_train)
    assert isinstance(model_test, xgb.sklearn.XGBClassifier), f'''Function train_model failed, it didn't return a right model'''


def test_inference(model, data):
    X_test = data[2]
    y_preds = inference(model, X_test)
    
    assert isinstance(y_preds,np.ndarray), f'''The predictions returned by inference function is not a numpy array'''
    assert y_preds.any() == 1, f'''The predictions seems to be wrong, cannot find any positive prediction'''


def test_compute_model_metrics(model, data):
    X_test = data[2]
    y_test = data[3]
    y_preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, y_preds)

    assert precision > 0.5, f"Precision is too low. precision = {precision} <= 0.5 "
    assert recall > 0.0, f"Recall value is wrong. recall = {recall} <= 0.0"
    assert fbeta > 0.0, f"fbeta is too low. fbeta = {fbeta} <= 0.5 "