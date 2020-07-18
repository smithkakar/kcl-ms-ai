#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

from sklearn.ensemble import RandomForestClassifier

from IPython.display import display

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn import metrics

from imblearn.ensemble import BalancedRandomForestClassifier
import xlsxwriter


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

relevant_indexes = []
demographics_column_indexes = ['Age', 'Gender', 'IUIPC-Awareness', 'IUIPC-Collection', 'IUIPC-Control',
                               'Online-Presence', 'Personal-Stability', 'Reciprocity']

relevant_indexes.extend(demographics_column_indexes)

"""[This function imputes missing data in any column with the mean value]

    Returns:
        [dataframe] -- [imputed data]
"""


def impute_data(data):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputed_data = pd.DataFrame(imp.fit_transform(
        data), columns=data.columns, index=data.index)

    return imputed_data


"""[This function transforms the data using MinMaxScaler]

    Returns:
        [dataframe] -- [transformed training data]
"""


def data_scale(data):
    scaler = MinMaxScaler()
    standard_data = pd.DataFrame(scaler.fit_transform(
        data), columns=data.columns, index=data.index)

    return scaler, standard_data


"""[This function transforms the data back to original form]

    Returns:
        [dataframe] -- [original data]
"""


def data_inverse_scale(scaler_object, data):
    inverse_scaled_data = scaler_object.inverse_transform(data)

    return inverse_scaled_data


range_n_clusters = list(range(2, 11))
results = {}
clusters = {}
important_features = {}
cluster_segments = {}

i = 0

writer = pd.ExcelWriter('K-Means_Clustering_Results.xlsx', engine='xlsxwriter')
workbook = writer.book
print('Running K-Means Clustering Algorithm.')
for question_number in range(1, 51):

    question = 'Q' + str(question_number).zfill(2)

    columns = []
    columns.extend([str(question_number).zfill(2) + '-Effort',
                    str(question_number).zfill(2) + '-Relevance',
                    str(question_number).zfill(2) + '-Uncomfortable',
                    str(question_number).zfill(2) + '-Truthfulness'])
    relevant_indexes.extend(columns)
    question_label = str(question_number).zfill(2) + '-Truthfulness'
    cleaned_reordered_user_responses = pd.DataFrame(
        reordered_user_responses[relevant_indexes].dropna()).reset_index(drop=True)

    train_x_question = cleaned_reordered_user_responses.copy()
    train_y_question = train_x_question.loc[:, question_label]
    train_x_question.drop(columns=question_label, inplace=True)

    scaler, good_data = data_scale(train_x_question)
    # Feature Selection
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(good_data, train_y_question)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    important_features[question] = indices[0:3].tolist()
    good_data = pd.DataFrame(data_inverse_scale(
        scaler, good_data), index=good_data.index, columns=good_data.columns)

    featured_train_data = good_data.iloc[:, important_features[question]]

    features = [relevant_indexes[i] for i in indices[0:3].tolist()]
    features = sorted(features)
    features.append(str(question_number).zfill(2) + '-Truthfulness')

    featured_train_data = featured_train_data.sort_index(axis=1)

    good_data = pd.concat([featured_train_data, train_y_question], axis=1)

    scaler, good_data = data_scale(good_data)
    # print('Running K-Means for ' + question + '.')
    clusterer = KMeans(n_clusters=2).fit(good_data)
    preds = clusterer.predict(good_data)

    centers = clusterer.cluster_centers_
    labels = clusterer.labels_

    true_centers = data_inverse_scale(scaler, centers)

    good_data['labels'] = list(map(int, labels))

    score = metrics.silhouette_score(good_data, preds)
    # print("For n_clusters = {}. The average silhouette_score with Kmeans is : {}".format(2, score))

    segments = ['Cluster {}'.format(i) for i in range(0, len(centers))]
    true_centers = pd.DataFrame(np.round(true_centers), columns=features)
    true_centers.index = segments
    # display(true_centers)

    cluster_segments[question] = true_centers

    true_centers.to_excel(writer, sheet_name='Sheet1', startrow=i, startcol=0)

    i = i+4
    del relevant_indexes[8:]
writer.save()
print('Saving Results to the current directory.')
