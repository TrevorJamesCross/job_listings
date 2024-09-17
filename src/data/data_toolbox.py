"""
Job Listings: Data Toolbox
Author: Trevor Cross
Last Updated: 09/16/24

Series of functions used to assist in manipulating raw data.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# -----------------------------------------
# ---Define Data Preprocessing Functions---
# -----------------------------------------


# define function to call OpenAI LLM to extract information
# from job description
def extract_features(string_description, openai_client, model_name):

    # define messages to model
    system_content = "You are an assistant that responds in Python \
            dictionary format. Return a Python dictionary with the \
            following keys: 'skills' and 'edu_level'. \
            Given a job description, extract all technical skills and tools \
            mentioned. Also extract the education level (one of associate's, \
            bachelor's, master's, or phd; take the lowest if more than one is \
            listed) if it present. If it is not present, replace those return \
            values with Python 'None' data type. Respond only with a \
            Python dictionary."

    messages = [
            {
                "role": "system",
                "content": system_content
                },
            {
                "role": "user",
                "content": string_description
                }
            ]

    # get LLM response
    response = openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0
            )
    content = response.choices[0].message.content

    try:
        # parse content into Python dict
        dict_content = eval(
                content.replace("```", "").replace("python", "").strip()
                )

        # lowercase content values
        dict_content['skills'] = [
                val.lower() for val in dict_content['skills']
                ]
        if isinstance(dict_content['edu_level'], str):
            dict_content['edu_level'] = dict_content['edu_level'].lower()
        else:
            dict_content['edu_level'] = "bachelor's"

        # return content
        return dict_content

    except Exception:
        # return dict of None vals if parsing fails
        # print(f"\nThe following description could not be parsed \
        #        {string_description}")
        return {
                'skills': None,
                'edu_level': None
                }


# define function to parse salary estimates using LLM
def parse_salary(string_salary, openai_client, model_name):

    # define messages to model
    system_content = "You are an assistant that parses salary ranges. \
            Return a single number that is the estimated annual salary. \
            If the range is given per hour, multiply the values by 200 \
            to get the yearly salary. Only return a single number which \
            is the average of the salary range. Never use the '$' sign."

    messages = [
            {
                "role": "system",
                "content": system_content
                },
            {
                "role": "user",
                "content": string_salary
                }
            ]

    # get LLM response
    response = openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0
            )
    content = response.choices[0].message.content

    return content
