"""
Job Listings: Preprocess
Author: Trevor Cross
Last Updated: 09/16/24

Preprocess raw job listings data using an LLM to parse out ML input features.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import standard libraries
import pandas as pd

# import openai
from openai import OpenAI

# import support libraries
import argparse
import os
import json

# import toolbox functions
from data_toolbox import (
        extract_features,
        parse_salary
        )

# -------------------
# ---Read Raw Data---
# -------------------

# define CLI argument parser
parser = argparse.ArgumentParser(description='read a CSV file from CLI')
parser.add_argument('csv_file', type=str, help='path to the input CSV file')

# parse the arguments
args = parser.parse_args()

# read the CSV file
df = pd.read_csv(args.csv_file)

# ----------------
# ---Clean Data---
# ----------------

# lowercase col names & replace spaces with underscores
df.columns = df.columns.str.lower().str.replace(" ", "_")

# retain relevant cols
df = df[
        [
            'job_description',
            'location',
            'size',
            'founded',
            'type_of_ownership',
            'industry',
            'sector',
            'revenue',
            'salary_estimate'
            ]
        ]

# remove rows w/ missingness (-1 represents missingness)
df = df[~df.isin([-1, '-1']).any(axis=1)]
df.reset_index(drop=True, inplace=True)

# ----------------------------------------------
# ---Parse Out Features from Job Descriptions---
# ----------------------------------------------

# define path to OpenAI key
key_path = os.path.join(".secrets", "openai_creds.json")
with open(key_path, 'r') as f:
    dict_creds = json.load(f)

# init OpenAI client
client = OpenAI(
        api_key=dict_creds['api_key'],
        organization=dict_creds['organization'],
        project=dict_creds['project']
        )

# define OpenAI model to use
model_name = 'gpt-4o-mini'

# parse salary_estimate
df['salary_estimate'] = df['salary_estimate'].apply(
        lambda x: parse_salary(x, client, model_name)
        )

# collect description features into list
list_dict = []
for desc in df['job_description']:
    list_dict.append(
            extract_features(desc, client, model_name)
            )

# create DataFrame from extracted job requirements
df_req = pd.DataFrame(list_dict)

# concat df w/ df_req
df = pd.concat([df, df_req], axis=1)

# drop cols & rows
df.drop(columns=['job_description'], inplace=True)
df = df[~df.isin([None, 'none' 'None']).any(axis=1)]
df.reset_index(drop=True, inplace=True)

# ------------------------------
# ---Output Preprocessed Data---
# ------------------------------

# define output directory
output_dir = os.path.join("data", "preprocessed")

# create output dir if not exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# output file
df.to_csv(
        output_dir + "/glassdoor_jobs.csv",
        header=True,
        index=False
        )
