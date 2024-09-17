"""
Job Listings: Model Toolbox
Author: Trevor Cross
Last Updated: 09/17/24

Series of functions used to assist w/ model development & training.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import sklearn libraries
from sklearn.preprocessing import MultiLabelBinarizer

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
