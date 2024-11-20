from fastapi import FastAPI, Query
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional 

import src.data_processing as dp
import src.models as ml


## instantiate the api and define the tags meatadata & security protocol
tags_metadata = [
    {'name': 'Public', 'description':'Methods available to everyone'},
    {'name': 'Users', 'description':'Methods available for registered user'}#,
    # {'name': 'Admins', 'description':'Methods available for administrators'}
]


api = FastAPI(
    title = "NBA players",
    description = "Predict if a player is likely to have a career longer than 5 years or not based on his game statistics. \n \
                   Evaluate the investment pertinence with the model confidence (tpr) and a user defined assumed risk (fpr)",
    version = '1.0.0',
    openapi_tags = tags_metadata
)



#################################################################################
############################# Login protocol ####################################
#################################################################################





#################################################################################
########################## Define Public Methods ################################
#################################################################################


###================================= app status ======================================###
@api.get('/app_status', name = 'Quick test if the API is running',
         tags = ['Public'])
async def get_app_status():
    return {'status' : 'App is available'}




#################################################################################
########################## Define Private Methods ###############################
#################################################################################



###============================= Get model prediction ===============================###

class Player(BaseModel):
    """Encapsulates a single player statistic data"""
    Name: str = Field(default='playerName', description="The player's name")
    GamesPlayed: int = Field(..., ge=0, description="Number of games played")
    MinutesPlayed: float = Field(..., ge=0, description="Minutes played per game")
    PointsPerGame: float = Field(..., ge=0, description="Points per game")
    FieldGoalsMade: float = Field(..., ge=0, description="Number of basket scored from the field, excluding free throws, per game")
    FieldGoalAttempts: float = Field(..., ge=0, description="Field goals attempts, per game")
    FieldGoalPercent: float = Field(..., ge=0, le=100, description="Percentage of field goals attemps that are successfull")
    ThreePointMade: float = Field(..., ge=0, description="Number of 3-points made from beyond the 3-point line, per game")
    ThreePointAttempts: float = Field(..., ge=0, description="Number of shots made from beyond the 3-point line, per game")
    ThreePointPercent: float = Field(..., ge=0, le=100, description="Percentage of 3-point attemps that are successfull")
    FreeThrowMade: float = Field(..., ge=0, description="Number of shots made from the free-throw line, per game")
    FreeThrowAttempts: float = Field(..., ge=0, description="Number of shots attempted from the free-throw line, per game")
    FreeThrowPercent: float = Field(..., ge=0, le=100, description="Percentage of free throws attempts that are sucessfull")
    OffensiveRebounds: float = Field(..., ge=0, description="Number of times,per game, that a player recovers the ball after a missed shot by their own team")
    DefensiveRebounds: float = Field(..., ge=0, description="Number of times,per game, that a player recovers the ball after a missed shot by the opponent team")
    Rebounds: float = Field(..., ge=0, description="Total number of rebouns a player collects (OREB + DREB)")
    Assists: float = Field(..., ge=0, description="Number of passes, by game, made to a teammate that resulting in points")
    Steals: float = Field(..., ge=0, description="Number of times per game, a player steels the ball from the opponent team.")
    Blocks: float = Field(..., ge=0, description="Number of times per game, a player stops an apponent attack")
    Turnovers: float = Field(..., ge=0, description="Number of times per game, a player lossses possesion of the ball to the opposing team")


# Predefined the available models a user can choose:
available_models = ['Best_Model', 'High_precision', 'High_recall']
Models = Enum('Models', {item: item for item in available_models}, type = str)


# Define endpoint to request a prediction by the model:
@api.post('/prediction', name = 'Get Model Prediction',
                         description = 'Predict for a  single player based on his stats data',
                         tags = ['Users'])
async def get_prediction(
                    player : Player,
                    user_model : Models = Query('Best_Model', 
                                                description="The model to be used to make predictions, default is 'Best'"),
                    risk : Optional[float] = Query(None, 
                                                   description="Admissible risk factor [0,1]. Default best value depends on the model.",
                                                   ge=0.0,
                                                   le=1.0),
                    ):

    default_risks = {
        'Best_Model' : 0.6128, 
        'High_precision' : 0.6417, 
        'High_recall' : 0.5968
        }
    
    if risk is None:
        risk = default_risks[user_model]

    
    models_names_dict = {'Best_Model' : 'best_model',
                         'High_precision' : 'high_precision_model',
                         'High_recall' : 'high_recall_model'}
    model = models_names_dict[user_model]

    predictions = get_new_model_prediction(player, risk, model)
    
    outcome_message, suggest_message = get_api_message(predictions,player.Name)
    
    # print formatted predictions
    return {
            'Prediction' : outcome_message,
            'Suggestion' : suggest_message,
            'predicted probability' : round(predictions['probability'],4),
            'tpr' : f"The model has {predictions['tpr']*100: .1f} % rate of detecting a good player",
            'fpr' : f"the model has {predictions['fpr']*100: .1f} % rate of suggesting a wrong player (risk)",
            'threshold probability' : round(predictions['threshold'],4)
            }
    


def get_new_model_prediction(player, risk, model):
    '''
    Process the endpoint input data and make a prediction with the selected model
    
    Parameters:
        player (Player): The player statistic passed to api "Get Model Prediction" method.
        risk (float): Accepted false positive rate (risk of investing in the wrong player).
        model (str): Name of the model to use for predictions.

    Returns:
        predictions (dict): Encapsulates the predicted class, probability, classification threshold, tpr and fpr.
    '''

    # format player stats into an array or dataframe    
    df_player = dp.make_dataset(player)
    
    # and data tranformation for model
    features = dp.transform_data(df_player, model)

    # get model
    trained_model = ml.get_trained_model(model)

    # get model probability predictions
    probability = ml.predict_probability(trained_model, features)

    # get probability threshold from desired risk (use ROC-AUC)
    threshold, risk_tpr, risk_fpr = ml.get_probability_threshold(risk, model)
    
    # compare probability prediction with threshold to define final prediction
    predicted_class = ml.get_predicted_class(probability, threshold)
    
    predictions = {
        'predicted_class' : predicted_class,  #pred_class
        'probability' : probability,          #probablity
        'threshold' : threshold,              #threshold
        'tpr' : risk_tpr,                     #risk_tpr
        'fpr' : risk_fpr                      #risk_fpr
    }
    
    return predictions


def get_api_message(predictions,player_name):
    """Define message to print as a function of the predicted class membership"""
    
    if predictions['predicted_class'] == 1:
        outcome_mssg = f'Player {player_name} is predicted to stay longer than 5 years.'
        suggest_mssg = 'Potentially a good investment'
    else:
        outcome_mssg = f'Player {player_name} is predicted to stay less than 5 years.'
        suggest_mssg = 'Less interesting or risky investment'
    
    return outcome_mssg, suggest_mssg