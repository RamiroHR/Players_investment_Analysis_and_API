# Author
Ramiro H. Rodriguez

# Version
Demo_1.3  


# Project description
Create a model that predicts if an NBA player will have a carrer longer than 5 years, making it a good investment opportunity.  
Create an api wrapping this model to assits users in deciding if investing in a particular player is a good decision or not.  

<br>


# Project Organization
    |
    ├── README.md                  <- The top-level README with a description of this project.
    |
    |
    ├── requirements.txt           <- The requirements file for reproducing the analysis environment venv.
    ├── venv/
    |
    |
    ├── data.csv                   <- Dataset: player's games statistics
    |
    |
    ├── Data_Exploration.ipynb     <- Notebook for data exploration and visualizations
    |
    ├── Modeling.ipynb             <- Notebook to explore, train, select and validate models
    │
    │
    ├── models/                    <- Trained and serialized models, model predictions and model indicators
    │
    ├── src                        <- Source code for use in this project.
    │   ├── __init__.py            <- Makes src a Python module
    │   │
    │   ├── data_processing.py     <- Functions and tools to preprocess data
    │   └── models.py              <- Functions for calling and using pre-trained models
    │
    │
    └── main.py                    <- API script

<br>

# Create the environment
Use the ```requirements.txt``` file to create the environment on the root folder.
For example, using _vortualenv_ in a terminal:
1. ```pip install virtualenv```  (if you don't already have virtualenv installed)
1. ```virtualenv venv``` to create the new environment named 'venv' within the project directory. 
1. ```source venv/bin/activate``` or ```source venv/Scripts/activate``` to activate the virtual environment.
1. ```pip install -r requirements.txt``` to install the required packages in the current environment.

<br>

# Data
The file ```data.csv``` contains labeled data of several players. Data consists in various games statistics of the player.

<br>

# Notebooks
The ```Data_Exploration.ipynb``` contains a first analysis of the raw data. This notebook helps to gain insights on the distribution of the data, the relation among the features and with the target variable. Insights gained during this phase helps to decide on the relevant features to consider and the models that could ahndle well the classification task.

The ```Modeling``` notebook explore the performance fo differents models and is used to select the best models to use in the API.

<br>

# Launch the API
In the API the user cna chose between 3 models available:  
* The **Best** model which for a given risk tries to maximize the the recall (model with the best auc score)
* The **High accuracy** model which keeps the precision high for an acceptable recall
* the **High recall** model which tries to reach a high recall, at the expenses of a lower precision.

For the 3 models the user can select a desired **risk** which account for the rate of false positive: the rate at which a less promizing player is mistakenly classified as a player that will have a career longer than 5 years, resulting in a bad investment.


To lunch the API in the localhost run the following comand in a terminal:  
```
uvicorn main:api --reload
```


