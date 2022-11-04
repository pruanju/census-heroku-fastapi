import json
import requests
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


url = 'http://udacity-heroku-fastapi-app.herokuapp.com/predict'
#Example for negative classification
data_0 = {
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

#Example for positive classification
data_1 = {
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


if __name__ == '__main__':
    response = requests.post(url, data=json.dumps(data_1))

    if response.status_code == 200:
        result = response.json()['Prediction']
        logging.info(f"The prediction for the provided data is: {result}")
    else:
        logging.error(
            f'Something went wrong, return code is {response.status_code}')