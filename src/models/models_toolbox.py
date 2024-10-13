"""
Job Listings: Model Toolbox
Author: Trevor Cross
Last Updated: 10/13/24

Series of functions used to assist w/ model development & training.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import sklearn libraries
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        mean_absolute_percentage_error,
        root_mean_squared_error
        )

# import system/file libraries
import json

# -------------------------------------
# ---Define Model Pipeline Functions---
# -------------------------------------


# define a function to preprocess "skills" col using MultiLabelBinarizer
def transform_skills(X, classes=None):

    # init multilabelbinarizer
    mlb = MultiLabelBinarizer(
            classes=classes
            )

    # return fitted binarizer
    return mlb.fit_transform(X)

# ---------------------------------------
# ---Define Model Evaluation Functions---
# ---------------------------------------


# define function to evaluate model across different metrics
def eval_metrics(true_values, predicted_values):
    return {
            metric: func(
                true_values,
                predicted_values
                ) for metric, func in zip(
                [
                    'MSE',
                    'RMSE',
                    'MAE',
                    'MAPE'
                    ],
                [
                    mean_squared_error,
                    root_mean_squared_error,
                    mean_absolute_error,
                    mean_absolute_percentage_error
                    ]
                )
            }

# {
#             'MSE': mean_squared_error(
#                 true_values,
#                 predicted_values,
#                 ),
#             'RMSE': root_mean_squared_error(
#                 true_values,
#                 predicted_values,
#                 ),
#             'MAE': mean_absolute_error(
#                 true_values,
#                 predicted_values
#                 ),
#             'MAPE': mean_absolute_percentage_error(
#                 true_values,
#                 predicted_values
#                 )
#             }
#
# ----------------------------------
# ---Define System/File Functions---
# ----------------------------------


# define function to save dictionary as JSON file
def dict_to_json(my_dict, file_path):
    with open(file_path, "w+") as file:
        json.dump(my_dict, file, indent=4)
