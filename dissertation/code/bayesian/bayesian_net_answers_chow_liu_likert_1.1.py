#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from pomegranate import BayesianNetwork
import pygraphviz

BAYESIAN_DIR = 'images/bayesian'

"""[This function saves the images to the hard disk]

Returns:
    [png] -- [png file]
"""


def save_fig(folder, fig_id, tight_layout=True):
    if not os.path.isdir(folder):
        os.makedirs(folder)

    file_path = os.path.join(folder, fig_id + '.png')
    if tight_layout:
        plt.tight_layout()
    plt.savefig(file_path, format='png', dpi=300)


"""[This function loads the data from csv file to pandas dataframe]

    Returns:
        [dataframe] -- [data]
"""


def load_survey_data():
    df = pd.read_csv('All_Responses_Removed.csv')
    return df


user_responses = load_survey_data()
user_responses.drop(columns=['Prolific ID'], inplace=True)
user_responses.columns = user_responses.columns.str.replace(r'[\s\n\t ]+', '-')
user_responses.columns = user_responses.columns.str.replace(r'[a-d]-', '-')
demographics_data = user_responses.iloc[:, :8]
demographics_user_responses = demographics_data.reindex(
    sorted(demographics_data.columns), axis=1)
question_subset = user_responses.reindex(
    sorted(user_responses.columns[8:]), axis=1)
reordered_user_responses = pd.concat(
    [demographics_user_responses, question_subset], axis=1)

labels = reordered_user_responses.iloc[:, 10:207:4]
labels.fillna(labels.mean(), inplace=True)
labels = labels.round(0)

blocks = {}
block_1 = [3, 4, 8, 12, 16, 17, 22, 25, 28, 30, 34, 37, 38, 40, 44, 48, 49]
block_2 = [2, 6, 10, 13, 14, 18, 19, 24, 27, 31, 32, 35, 39, 41, 45, 47, 50]
block_3 = [1, 5, 7, 9, 11, 15, 20, 21, 23, 26, 29, 33, 36, 42, 43, 46]
column_names_1 = ['DoB', 'Birth Country', 'Home Address', 'Workplace Travel', 'Ethnicity', 'Politics',
                  'Memorable Event', 'Study/Education', 'Alcoholic Beverages', 'Favourite Mobile Brand',
                  'Author', 'Government Banned Movie', 'Web Browser', 'Favourite Actor/Actress',
                  'Last movie at cinema', 'Go to cinema with', 'Rent adult movies']
column_names_2 = ['Gender', 'City of Residence', 'Workplace Postcode', 'Personal Email',
                  'Professional Email', 'Religion', 'Sexual Orientation', 'Illnesses',
                  'Hobby/Pastime', 'Hurt Sentiments - Movie', 'Holiday Destination', 'Music Genre',
                  'Age for Adult movie', 'Favourite Movie', 'Money on cinema weekly',
                  'Illegal streaming/downloading', 'Favourite Pornstar']
column_names_3 = ['Name', 'Country of Residence', 'Home Postcode', 'Employer Name', 'Work Address',
                  'Phone Number', 'Relationship Status', 'Lied to Partner', 'Languages',
                  'Annual Income', 'Shared X-rated movies', 'Lied about Age', 'Musician',
                  'Favourite Movie Genre', 'Favourite Soundtrack', 'Online rental subscriptions']

blocks = dict(zip(block_1, column_names_1))
blocks.update(dict(zip(block_2, column_names_2)))
blocks.update(dict(zip(block_3, column_names_3)))
column_names = []
for k, v in sorted(blocks.items()):
    column_names.append(v)

fig = plt.figure(figsize=(30, 30), dpi=300)
bayes_algorithm = 'chow-liu'
bayesian_net_models = {}
print('Generating Chow-Liu Tree for Truthfulness across questions.')
model = BayesianNetwork.from_samples(
    labels, state_names=column_names, algorithm=bayes_algorithm, n_jobs=-1)
plt.title('Truthfulness \n Bayesian Network \n',
          fontsize=30, fontweight='bold')
model.plot(with_labels=True)

save_fig(BAYESIAN_DIR, bayes_algorithm + '_full_bayesian_net_likert')
print('Saving Chow-Liu Tree for Truthfulness across questions in ' +
      BAYESIAN_DIR + ' directory.')