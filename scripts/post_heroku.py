import json
import requests
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


url = 'http://udacity-heroku-fastapi-app.herokuapp.com/'
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
# , 'accept': 'application/json'}
headers = {'content-type': 'application/json'}

if __name__ == '__main__':
    # , headers=headers)
    response = requests.post(url, data=json.dumps(payload), headers=headers)

    if response.status_code == 200:
        result = response.json()['result']
        logging.info(f"The prediction for given sample is: {result}")
    else:
        logging.error(
            f'Something went wrong, response code is {response.status_code}')