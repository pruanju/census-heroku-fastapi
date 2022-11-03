import numpy as np
import sys

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split

sys.path.append('./starter/ml')
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


    

def test_train_model(model, encoder, lb):
    assert isinstance(model, XGBClassifier), f'''Function train_model failed, it didn't return a right model'''
    assert isinstance(encoder, OneHotEncoder), f'''Function train_model failed, it didn't return a right feature encoder'''
    assert isinstance(lb, LabelBinarizer), f'''Function train_model failed, it didn't return a right label encoder'''


def test_inference(model, encoder, lb, dataframe):
    traindata, testdata = train_test_split(dataframe, test_size=0.20)
    X_test, y_test, _, _ = process_data(testdata, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)
 
    #X_test = data_for_testing[0]
    y_preds = inference(model, X_test)
    
    assert isinstance(y_preds,np.ndarray), f'''The predictions returned by inference function is not a numpy array'''
    assert y_preds.any() == 1, f'''The predictions seems to be wrong, cannot find any positive prediction'''


def test_compute_model_metrics(model, encoder, lb, dataframe):
    traindata, testdata = train_test_split(dataframe, test_size=0.20)
    X_test, y_test, _, _ = process_data(testdata, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)
 
    #X_test = data_for_testing[0]
    #y_test = data_for_testing[1]
    y_preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, y_preds)

    assert precision > 0.5, f"Precision is too low. precision = {precision} <= 0.5 "
    assert recall > 0.5, f"Recall value is wrong. recall = {recall} <= 0.5"
    assert fbeta > 0.5, f"fbeta is too low. fbeta = {fbeta} <= 0.5 "