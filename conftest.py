import pytest
import pandas as pd
import sys
import pickle



sys.path.append('./starter/ml')


@pytest.fixture(scope='session')
def dataframe(request):
    dataframe = pd.read_csv("./data/census.csv")
    return dataframe


@pytest.fixture(scope='session')
def model(request):
    model = pickle.load(open('./model/Census_model.pickle', 'rb'))
    return model


@pytest.fixture(scope='session')
def lb(request):
    lb = pickle.load(open('./model/Label_model.pickle', 'rb'))
    return lb


@pytest.fixture(scope='session')
def encoder(request):
    encoder = pickle.load(open('./model/Encoder_model.pickle', 'rb'))
    return encoder


@pytest.fixture(scope='session')
def json_sample_1():
    payload = {
        'age': 25,
        'workclass': 'Private',
        'fnlgt': 200681,
        'education': 'Some-college',
        'education-num': 10,
        'marital-status': 'Never-married',
        'occupation': 'Prof-speciality',
        'relationship': 'Own-child',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 40,
        'native-country': 'United-States',
        'salary': '<=50K'
    } 
    return payload

@pytest.fixture(scope='session')
def json_sample_2():
    payload = {
        'age': 34,
        'workclass': 'Private',
        'fnlgt': 202822,
        'education': 'Doctorate',
        'education-num': 14,
        'marital-status': 'Never-married',
        'occupation': 'Prof-speciality',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Female',
        'capital-gain': 14000,
        'capital-loss': 0,
        'hours-per-week': 60,
        'native-country': 'United-States',
        'salary': '>50K'
    } 
    return payload


@pytest.fixture(scope='session')
def json_sample_with_error():
    payload = {
        'age': 34,
        'workclass': 'Private',
        'fnlgt': 202822,
        'education': 'Doctorate',
        'education-num': 14,
        'race': 'White',
        'sex': 'Female',
        'capital-loss': 0,
        'hours-per-week': 60,
        'native-country': 'United-States',
        'salary': '>50K'
    } 
    return payload


