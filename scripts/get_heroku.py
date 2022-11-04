import requests
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


url = 'http://udacity-heroku-fastapi-app.herokuapp.com/'

if __name__ == '__main__':
    response = requests.get(url)

    if response.status_code == 200:
        result = response.json()
        logging.info(f"{result}")
    else:
        logging.error(
            f'Something went wrong, response code is {response.status_code}')