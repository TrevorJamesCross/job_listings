"""
Job Listings: Train Model
Author: Trevor Cross
Last Updated: 09/17/24

Develop & tune model pipeline on preprocessed data.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import standard libraries
import numpy as np
import pandas as pd

# import scikit-learn libraries
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
        train_test_split,
        GridSearchCV
        )
from sklearn.preprocessing import (
        OrdinalEncoder,
        OneHotEncoder,
        FunctionTransformer
        )
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor

# import support libraries
import argparse
from tempfile import TemporaryDirectory
from joblib import Memory

# import toolbox functions
from models_toolbox import transform_skills

# ------------------------------------
# ---Read & Split Preprocessed Data---
# ------------------------------------

# print progress to CLI
print("\nReading & splitting input data...")

# define CLI argument parser
parser = argparse.ArgumentParser(description='read a CSV file from CLI')
parser.add_argument('csv_file', type=str, help='path to the input CSV file')

# parse the arguments
args = parser.parse_args()

# read the CSV file & pop target col
df = pd.read_csv(args.csv_file)
y = df.pop('salary_estimate')

# evaluate skills col vals to lists
df['skills'] = df['skills'].apply(lambda x: eval(x))

# get list of all possible categores for skill col
all_categories = set([cat for sublist in df['skills'] for cat in sublist])
all_categories = list(all_categories)

# split data
X_trn, X_tst, y_trn, y_tst = train_test_split(
        df,
        y,
        test_size=0.20,
        random_state=42
        )

# -----------------------
# ---Build Transformer---
# -----------------------

# print progress to CLI
print("\nBuilding model pipeline transformer...")

# define lists of feature categories
ord_cols = ['size', 'revenue', 'edu_level']
oh_cols = ['location', 'type_of_ownership', 'industry', 'sector']
skills_col = ['skills']
passed_cols = ['founded']

# define lists of ordinal encodings in order
size_ord = [
        'Unknown',
        '1 to 50 Employees',
        '51 to 200 Employees',
        '201 to 500 Employees',
        '501 to 1000 Employees',
        '1001 to 5000 Employees',
        '5001 to 10000 Employees',
        '10000+ Employees'
        ]
revenue_ord = [
        'Unknown / Non-Applicable',
        'Less than $1 million (USD)',
        '$1 to $5 million (USD)',
        '$5 to $25 million (USD)',
        '$25 to $100 million (USD)',
        '$100 to $500 million (USD)',
        '$500 million to $1 billion (USD)',
        '$1 to $5 billion (USD)',
        '$5 to $10 billion (USD)',
        '$10+ billion (USD)'
        ]
edu_ord = [
        'none',
        "associate's",
        "bachelor's",
        "master's",
        'phd'
        ]

# init ordinal encoder
ord_enc = OrdinalEncoder(
        categories=[size_ord, revenue_ord, edu_ord]
        )

# init one-hot encoder
oh_enc = OneHotEncoder(
        handle_unknown='infrequent_if_exist'
        )

# define label binarizer as FunctionTransformer
mlb_trans = FunctionTransformer(
        transform_skills,
        validate=False,
        kw_args={'classes': all_categories}
        )

# define col transformer
preproc = ColumnTransformer(
        transformers=[
            ('ord_enc', ord_enc, ord_cols),
            ('oh_enc', oh_enc, oh_cols),
            ('mlb', mlb_trans, skills_col[0])
            ],
        remainder='passthrough'
        )

# ---------------------------------
# ---Build & Tune Model Pipeline---
# ---------------------------------

# print progress to CLI
print("\nBuilding & tuning model pipeline...")

# init model
model = GradientBoostingRegressor(
        loss='squared_error'
        )

# use temporary directory for memory caching
with TemporaryDirectory() as temp_dir:

    # init memory
    memory = Memory(location=temp_dir, verbose=0)

    # define model pipeline
    pipe = Pipeline(
            steps=[
                ('preprocessor', preproc),
                ('estimator', model)
                ],
            memory=memory
            )

    # define search grid params
    param_grid = {
            'estimator__n_estimators': np.arange(100, 600, 400),
            'estimator__learning_rate': np.arange(0.05, 0.25, 0.15),
            }
    grid = GridSearchCV(
            pipe,
            param_grid,
            scoring='neg_mean_squared_error',
            refit=True,
            n_jobs=-1,
            cv=2
            )

    # tune model
    grid.fit(X_trn, y_trn)

# --------------------
# ---Evaluate Model---
# --------------------

# print progress to CLI
print("\nEvaluating best model...")

# get best model
best_model = grid.best_estimator_

# evaluate training data

# evaluate test data

# ----------------------------
# ---Output Model & Metrics---
# ----------------------------

# print progress to CLI
print("\nOutputting model & metrics...")

# define artifacts dir path

# output model

# output metrics
