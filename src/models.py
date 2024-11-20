import pandas as pd
from joblib import load

model_path = './models/'


def get_trained_model(model_name):
    '''
    Loads a pre-trained model.
    
    Parameters:
        model_name (str): The name of the model
    
    Returns:
        object: The loaded pre-trained model
    '''    
    
    trained_model = load(model_path + model_name + '.joblib')
    return trained_model


def predict_probability(model, data):
    '''
    Predicts the probability of being classified as a postivie class (1) observation
    
    Parameters:
        model (object): pre-trained model.
        data (ndarray): 2D array of shape (1,n_features) containing the preprocessed data of a single player
    
    Returns:
        pos_class_probability (float): probability of belonging to the positive class (1)
    '''
    probability = model.predict_proba(data)
    pos_class_probability = probability[0][1]
    
    return pos_class_probability


def get_probability_threshold(risk, model_name):
    '''
    Finds the probability threshold to match a desired risk of false positive rate using the roc-auc performance of a model.
    
    Parameters:
        risk (float): accepted false positive rate
        model_name (str): name of the model being use.
    
    Returns:
        threshold (float): probability threshold above which an observation is classified as a member of the positive class.
        risk_tpr (float): resulting true positive rate for this threshold.
        risk_fpr (float): resulting false positive rate admitted by this threshold.
    '''    
    
    # load the roc auc data of the trained model. Cols: tpr, fpr, thresholds
    df = pd.read_csv(model_path + model_name + '_rocauc.csv')
 
    idx = abs(df.fpr-risk).idxmin()
 
    risk_tpr = df.tpr.loc[idx]
    risk_fpr = df.fpr.loc[idx]
    threshold = df.thresholds.loc[idx]
    
    return threshold, risk_tpr, risk_fpr

    

def get_predicted_class(probability, threshold):
    """Compare predicted probability with the probability threshold and make classification"""
    return 1 if probability > threshold else 0