import pandas as pd
import numpy as np
from joblib import load


def make_dataset(player): 
    '''
    Format a single player data input into a dataframe
    
    Parameters:
        player (Player): The player statistic passed to api "Get Model Prediction" method.
        
    Returns:
        player_df (Dataframe): player's statistic with columns names as used to train the model.
    '''
    
    player_data = {'Name': player.Name,
                'GP': player.GamesPlayed,
                'MIN': player.MinutesPlayed,
                'PTS': player.PointsPerGame,
                'FGM': player.FieldGoalsMade,
                'FGA': player.FieldGoalAttempts,
                'FG%': player.FieldGoalPercent,
                '3P Made': player.ThreePointMade,
                '3PA' : player.ThreePointAttempts,
                '3P%' : player.ThreePointPercent,
                'FTM' : player.FreeThrowMade,
                'FTA' : player.FreeThrowAttempts,
                'FT%' : player.FreeThrowPercent,
                'OREB' : player.OffensiveRebounds,
                'DREB' : player.DefensiveRebounds,
                'REB' : player.Rebounds,
                'AST' : player.Assists,
                'STL' : player.Steals,
                'BLK' : player.Blocks,
                'TOV' : player.Turnovers
                } 
    
    player_df = pd.DataFrame(player_data, index=[0])
    
    return player_df
    
    
def transform_data(df_player, model_name):
    '''
    Preprocess data: remove less relevant features, transform to array, scale features.
    
    Parameters:
        df_player (dataframe): single player data.
        
    Returns:
        X_data (ndarray): 2D array of shape (1,n_features) containing data of a single player
    '''
    
    # log transformation
    exclude_columns = ['Name','GP','FT%','TARGET_5Yrs']
    columns_to_transform = df_player.columns.difference(exclude_columns)
    df_player[columns_to_transform] = df_player[columns_to_transform].apply(np.log1p)        
    
    # drop irrelevant features
    columns_to_drop = {'best_model':[],
                       'high_precision_model':['3P Made', '3PA'],
                       'high_recall_model':[]}
    df_player.drop(columns_to_drop[model_name], axis=1, inplace=True)
    
    # dataframe to array
    X_data_vals = df_player.drop('Name', axis=1).values
    
    # normalize data
    scaler = load('./models/scaler_for_'+model_name+'.joblib')
    X_data = scaler.transform(X_data_vals)
    
    return X_data


    
    
    
    