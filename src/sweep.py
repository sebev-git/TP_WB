import os
import wandb
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import joblib

X_train = pd.read_csv('data/preprocessed/X_train.csv')
X_test = pd.read_csv('data/preprocessed/X_test.csv')
y_train = pd.read_csv('data/preprocessed/y_train.csv')
y_test = pd.read_csv('data/preprocessed/y_test.csv')
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# 1. Pick a method
sweep_config = {
    'method': 'random'
    }

# 2. Name hyperparameters
parameters_dict = {
    'criterion': {
        'values': ['gini', 'entropy', 'log_loss']
        },
    'splitter': {
        'values': ['best', 'random']
        },
    'max_depth': {
          'values': [None, 10, 20, 50, 100, 200, 500]
        },
    'random_state': {
        'values': [42]
    }
    }

sweep_config['parameters'] = parameters_dict

# 3. Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="classification-car-accidents")


def train(parameters=None):

    run = wandb.init(
        project="classification-car-accidents",
        tags=["sweep", "Decision Tree"],
        config=parameters
    )

    parameters = wandb.config

    dt_classifier = DecisionTreeClassifier(**parameters)
    dt_classifier.fit(X_train, y_train)

    train_accuracy = dt_classifier.score(X_train, y_train)
    test_accuracy = dt_classifier.score(X_test, y_test)
    wandb.log({"train_accuracy": train_accuracy, "test_accuracy": test_accuracy})

    wandb.finish()

# 5. Run the sweep agent
wandb.agent(sweep_id, train, count=5)