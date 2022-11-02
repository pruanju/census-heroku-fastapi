from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train, tuning=False):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    tuning: bool
        by default we don't perform tuning of hyperparameters (False)
    Returns
    -------
    model
        Trained machine learning model.
    
    """

    xgboost=XGBClassifier()
    if(tuning):
        parm_grid = { 'max_depth': [3,6,10],
           'subsample': [0.5, 0.8],
           'learning_rate': [0.01, 0.05, 0.1],
           'n_estimators': [200, 500, 700],
           'colsample_bytree': [0.3, 0.7]}
        #With above parameters will train the model.
        gridsearch = GridSearchCV(xgboost, param_grid = parm_grid , cv=5)
        gridsearch.fit(X_train,y_train)
        # We select the best parameters of the training
        best_parameters = gridsearch.best_params_
        #And we traing the model with the best parameters
        xgboost_model=XGBClassifier(**best_parameters)
        xgboost_model.fit(X_train,y_train)
    else:
        xgboost_model=XGBClassifier()
        xgboost_model.fit(X_train,y_train)
        
    
    return xgboost_model



def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : XGBoost
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    y_pred = model.predict(X)
    return y_pred




