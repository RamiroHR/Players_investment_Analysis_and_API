from fastapi import FastAPI, Query
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional 


## instantiate the api and define the tags meatadata & security protocol
tags_metadata = [
    {'name': 'Public', 'description':'Methods available to everyone'},
    {'name': 'Users', 'description':'Methods available for registered user'}#,
    # {'name': 'Admins', 'description':'Methods available for administrators'}
]


api = FastAPI(
    title = "NBA players",
    description = "Predict if a player will have career longer than 5 years",
    version = '1.0.0',
    openapi_tags = tags_metadata
)


model_path = '.\ ' 



#################################################################################
############################# Login protocol ####################################
#################################################################################





#################################################################################
########################## Define Public Methods ################################
#################################################################################


###========================= app status =========================###
@api.get('/app_status', name = 'Quick test if the API is running',
                    tags = ['Public'])
async def get_app_status():
    return {'status' : 'App is available'}


# @api.get('/')
# def get_index():
#     return {'data': 'hello world'}
    
# @api.get('/item/{itemid}')
# def get_item(itemid):
#     return {
#         "route" : "dynamic", 
#         "itemid": itemid
#     }

# @api.get('/')
# def get_index(argument1):
#     return {
#         'data': argument1
#     }
  
# @api.get('/typed')
# def get_typed(argument1: int):
#     return {
#         'data': argument1 + 1
#     }
    


#========================== Get model prediction (singular) ==========================###

class Player(BaseModel):
    # itemid: int
    # description: str
    # owner: Optional[str] = None
    Name: str = Field(default='playerName', description="The player name")
    GamesPlayed: int = Field(..., ge=0, description="The player name")
    MinutesPlayed: float = Field(..., ge=0)
    PointsPerGame: float = Field(..., ge=0)
    FieldGoalsMade: float = Field(..., ge=0)
    FleldGoalAttempts: float = Field(..., ge=0)
    FieldGoalPercent: float = Field(..., ge=0, le=100)
    # _3PointMade: float
    # _3PointAttempts: float
    # _3PointPercent: float
    # FreeThrowMade: float
    # FreeThrowAttempts: float
    # FreeThrowPercent: float
    # OffensiveRebounds: float
    # DefensiveRebcunds: float
    # Rebounds: float
    # Assists: float
    # Steals: float
    # Blocks: float
    # Turnovers: float


# Predefined 'assets' present in the database.
available_models = ['Null', 'Best']
Models = Enum('Models', {item: item for item in available_models}, type = str)

@api.post('/prediction', name = 'Get Model Prediction',
                         description = 'Get the prediction for a single player base on his stats data. Return a prediction, the investment potential(tpr) and the assumed risk (fpr)',
                         tags = ['Users'])

async def get_prediction(
                    player : Player,
                    risk : float = Query(0.3368, description="The risk factor, default is 0.3368"),
                    model : Models = Query("Best", description="The model to be used for prediction, default is 'Best'"),
                    ):

    predictions = get_new_model_prediction(player, risk, model)
        
    # print formatted predictions
    return {
            'Prediction' : f'Player {player.Name} has high potential',
            'tpr' : f"{predictions['probability']*100: .1f} % probability of investing in a good player",
            'fpr' : f"{predictions['threshold']*100: .1f} % probability of investing in a wrong player (risk)",
            'predicted prob' : predictions['tpr'],
            'threshold proba' : predictions['tpr']
            }
    

    
def get_new_model_prediction(player, risk, model):

    # # format player stats into an array or dataframe    
    # df_player = format_player_data(player)
    
    # # and data tranformation for model
    # features = transform_data(df_player)

    # # get model
    # trained_model = get_trained_model(model)

    # # get model probability predictions
    # probability = predict_probability(trained_model, features)

    # # get probability threshold from desired risk (use ROC-AUC)
    # threshold, risk_tpr, risk_fpr = get_probability_threshold(risk, model)
    
    # # compare prob prediction with threshold to define final prediction
    # pred_class = get_predicted_class(probability, threshold)
    
    predictions = {
        'predicted_class' : 1,  #pred_class
        'probability' : 0.8,   #probablity
        'threshold' : 0.6,     #threshold
        'tpr' : 0.77,          #risk_tpr
        'fpr' : 0.33           #risk_fpr
    }
    
    return predictions

    