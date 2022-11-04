# Put the code for your API here.
from typing import Optional
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
import pickle

from starter.ml.model import inference
from starter.ml.data import process_data



# Alias Generator function
def replace_dash(string: str) -> str:
    return string.replace('_','-')

# Definition of the data that will be provided to the POST requests
class CensusData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int 
    marital_status: str 
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int 
    capital_loss: int 
    hours_per_week: int 
    native_country: str 
    salary: Optional[str]

    class Config:
        alias_generator = replace_dash

# Instance the app. 
app = FastAPI() 



# Este metodo reemplaza a los 3 siguientes para que lo cargue todo en el arranque y no en la predicción!
@app.on_event("startup")
async def startup_event(): 
    global model, encoder, lb
    model = pickle.load(open("./model/Census_model.pickle", "rb"))
    encoder = pickle.load(open("./model/Encoder_model.pickle", "rb"))
    lb = pickle.load(open("./model/Label_model.pickle", "rb"))
"""    
# load the model, encoder and label from disk
model = pickle.load(open('./model/Census_model.pickle', 'rb'))
encoder = pickle.load(open('./model/Encoder_model.pickle', 'rb'))
lb = pickle.load(open('./model/Label_model.pickle', 'rb'))    
"""   
    
# Define a GET method. 
@app.get("/") 
async def say_hello(): 
    return {"Greetings!": "Hi welcome to this Udacity project!"} 


# POST request to /predict site. Used to validate model with sample census data
@app.post('/predict')
async def predict(input: CensusData):
    """
    POST request that will provide census data and return a prediction
    Output: 0 or 1
    """

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

    # Read data sent as POST
    input_data = input.dict(by_alias=True)
    input_df = pd.DataFrame(input_data, index=[0])


    # Process the data
    X, _, _, _ = process_data(input_df, categorical_features=cat_features, label='salary', training=False, encoder=encoder, lb=lb)

    preds = int(inference(model, X)[0])
    return {"Prediction": preds}

"""
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
"""





