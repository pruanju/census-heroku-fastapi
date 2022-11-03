from fastapi.testclient import TestClient

import pytest
import sys

sys.path.append('../')
from main import app

json_sample = {
    'age': 31,
    'workclass': 'Private',
    'fnlgt': 45781,
    'education': 'Masters',
    'education-num': 14,
    'marital-status': 'Never-married',
    'occupation': 'Prof-speciality',
    'relationship': 'Not-in-family',
    'race': 'White',
    'sex': 'Female',
    'capital-gain': 14000,
    'capital-loss': 0,
    'hours-per-week': 55,
    'native-country': 'United-States',
    'salary': '>50K'
}

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

        
@pytest.fixture
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

@pytest.fixture
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


@pytest.fixture
def json_sample_with_error():
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
        'capital-loss': 0,
        'hours-per-week': 60,
        'native-country': 'United-States',
        'salary': '>50K'
    } 
    return payload


def test_get_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "Greetings!": "Hi welcome to this Udacity project!"}


def test_predict_1(client, json_sample_1):
    response = client.post("/predict", json=json_sample_1)
    print("Hola caracola")
    print(response)
    print("Hola caraculo")
    assert response.status_code == 200
    assert response.json()['Prediction'] == 0


def test_predict_0(client, json_sample_2):
    response = client.post("/predict", json=json_sample_2)
    assert response.status_code == 200
    assert response.json()['Prediction'] == 1

def test_predict_error_422(client, json_sample_with_error):
    response = client.post("/predict", json=json_sample_with_error)
    assert response.status_code == 422