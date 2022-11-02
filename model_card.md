# Model Card
- Author: Juan Manuel Pruaño
- Last updated: Nov 2022

## Model Details

Algorithm used is XGBoost with hyperparameters tuning. 

The pipeline of the model is:
1. OnehotEncoder for the catagorical features
2. Label binarizer for the labels (the positive class '>50K' is encoded as 1 and '<=50K' is encoded as 0)
3. XGBoost algorithm with hyperparmeters tuning.
4. Train the XGBoost model wiht the best hyperparameters.



## Intended Use
That is a trianing project for Udacity nano degree program.
Model is trained to classify if a person belongs to a group earning above or below $50K, based on census data from various countries. 


## Training Data
Dataset is census data (https://archive.ics.uci.edu/ml/datasets/census+income) with  32563 records. This data is provided with the project under `data/census.csv` folder.

The model is trained on 80% of dataset, 20% is reserved for validation.


## Evaluation Data
20% of the census data is used to validate the model


## Metrics

Metrics are stored in `starter/metrics/` folder.
fbeta metric is used for this project. Overall performance of the model is fbeta=0.71 


## Ethical Considerations
Census data are data collected from people therefore the answers depend on how these people would like themselves to be seen and represented, and also depend on how much people want to reveal about themselves, so the data used to train the model might contain biases. 

Some attributes that are collected from a survey, like for example hours per week, might contain biases based on influence from co-workers. Not all countries are represented in the native country variable and the data set is probably not large enough to assume that the model predicts well with native country variable.

## Caveats and Recommendations
As this project was about CI/CD, more emphasis was put on having the project utilizes tools like GitHub and Heroku than on how well to configure the model, it was trained without too much effort. The model would also perform better with a larger data set.