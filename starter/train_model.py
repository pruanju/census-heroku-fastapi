# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import pickle
from pathlib import Path
import logging

from ml.data import process_data
from ml.model import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s- %(message)s")
logger = logging.getLogger()

logger.info("Loading census data..")

# Add code to load in the data.
try:
    data = pd.read_csv("../data/census.csv")
except FileNotFoundError:
        logger.error("Failed to load census.csv file")


# Optional enhancement, use K-fold cross validation instead of a train-test split.
logger.info("Splitting the data..")
train, test = train_test_split(data, test_size=0.20)


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

logger.info("Processing the data..")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)


# Train and save the best model, the encoder and the label, set tuning to True if you want to perform hyperparameters tuning
logger.info("Training the model, this would take several minutes due to the hyperparameters tuning!")
model = train_model(X_train, y_train, tuning=False)

logger.info("Saving the model, the encoder and the lable to disk..")
with open('../model/Census_model.pickle', 'wb') as f:
    pickle.dump(model, f)

with open('../model/Encoder_model.pickle', 'wb') as f:
    pickle.dump(encoder, f)
    
with open('../model/Label_model.pickle', 'wb') as f:
    pickle.dump(lb, f)

# Calculating metrics of the model and saving them to model_ouput.txt file in the model folder
logger.info("Calculating metrics of the model: precission, recall, ROC curve")
y_preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_preds)
with open('../model/model_output.txt', 'w') as f:
    f.write('Performance of the model:\n')
    f.write('------------------------------------------------------------------------------------------------------------\n')
    f.write("".join(['precission: ', str(round(precision,4)), '\t', 'recall: ', str(round(recall,4)), '\t', 'fbeta: ', str(round(fbeta,4))]))
    f.write("\n")


# Calculating the metrics for each categorical variable (Slices) and saving them to slices_output.txt file in the model folder
logger.info("Testing model performance on slices and save them to ../model/slices_output.txt")
metrics=[]
for cat in cat_features:
    logger.info(f"Calculating metrics for category: {cat}")
    cat_unique_values = test[cat].unique()
    for value in cat_unique_values:
        X_cat, y_cat, _, _ = process_data(
            test[test[cat] == value], categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
        )
        
        y_preds = inference(model, X_cat)
        precision, recall, fbeta = compute_model_metrics(y_cat, y_preds)
        metrics.append((cat, value, str(round(precision,4)), str(round(recall,4)), str(round(fbeta,4))))
            
with open('../model/slices_output.txt', 'w') as f:
    f.write(f"{metrics}\n")

