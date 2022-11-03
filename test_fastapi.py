from fastapi.testclient import TestClient
import pytest
import sys

sys.path.append('../')
from main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_get_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "Greetings!": "Hi welcome to this Udacity project!"}


def test_predict_1(client, json_sample_1):
    response = client.post("/predict", json=json_sample_1)
    assert response.status_code == 200
    assert response.json()['Prediction'] == 0


def test_predict_0(client, json_sample_2):
    response = client.post("/predict", json=json_sample_2)
    assert response.status_code == 200
    assert response.json()['Prediction'] == 1

def test_predict_error_422(client, json_sample_with_error):
    response = client.post("/predict", json=json_sample_with_error)
    assert response.status_code == 422