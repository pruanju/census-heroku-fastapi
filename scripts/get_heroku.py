import requests
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


url = 'http://udacity-heroku-fastapi-app.herokuapp.com/'
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
# , 'accept': 'application/json'}
headers = {'content-type': 'application/json'}

if __name__ == '__main__':
    # , headers=headers)
    response = requests.get(url)

    if response.status_code == 200:
        result = response.json()
        logging.info(f"The prediction for given sample is: {result}")
    else:
        logging.error(
            f'Something went wrong, response code is {response.status_code}')