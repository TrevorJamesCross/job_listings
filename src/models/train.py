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
import pandas as pd

# import scikit-learn libraries
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
        train_test_split,
        GridSearchCV
        )
from sklearn.preprocessing import (
        OrdinalEncoder,
        OneHotEncoder
        )
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor

# import support libraries
import argparse
# from tempfile import TemporaryDirectory

# import toolbox functions
# from data_toolbox import

# ------------------------------------
# ---Read & Split Preprocessed Data---
# ------------------------------------

# define CLI argument parser
parser = argparse.ArgumentParser(description='read a CSV file from CLI')
parser.add_argument('csv_file', type=str, help='path to the input CSV file')

# parse the arguments
args = parser.parse_args()

# read the CSV file & pop target col
df = pd.read_csv(args.csv_file)
for col in df.columns:
    print(col)
    print(df[col].unique())
    print()
raise
y = df.pop('salary_estimate')

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

# define lists of feature categories
ord_cols = ['size', 'revenue', 'edu_level']
oh_cols = ['location', 'type_of_ownership', 'industry', 'sector']
skill_col = ['skill']
passed_cols = ['founded']

# define lists of ordinal encodings in order
size_ord = [
        'Unknown',
        '1 to 50 Employees',
        '51 to 200 Employees',
        '201 to 500 Employees',
        '501 to 1000 Employees',
        '1001 to 5000 Employees',
        '5001 to 10000 Employees'
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
        columns=[size_ord, revenue_ord, edu_ord]
        )

# init one-hot encoder
oh_enc = OneHotEncoder()

# define col transformer
preproc = ColumnTransformer(
        transformers=[
            ('ord_enc', ord_enc, ord_cols),
            ('oh_enc', oh_enc, oh_cols)
            ],
        remainder='passthrough'
        )

# ---------------------------------
# ---Build & Tune Model Pipeline---
# ---------------------------------

# init model
model = GradientBoostingRegressor(
        loss='squared_error'
        )

# define model pipeline
pipe = Pipeline(
        steps=[
            ('preprocessor', preproc),
            ('estimator', model)
            ]
        )

# define search grid params
param_grid = {
        'estimator__n_estimators': [],
        'estimator__learning_rate': [],
        }
grid = GridSearchCV(
        pipe,
        param_grid,
        scoring='neg_mean_squared_error',
        refit=False,
        n_jobs=-1,
        cv=7
        )
