#!/usr/bin/env python3
# coding: utf-8

import os
import numpy as np
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

from sklearn.model_selection import GridSearchCV

from imblearn.ensemble import BalancedRandomForestClassifier

from sklearn.utils import class_weight

import warnings

warnings.filterwarnings('ignore')

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

RESULTS_DIR = 'results'

# The code for EstimatorSelectionHelper class is referred from 
# http://www.davidsbatista.net/blog/2018/02/23/model_optimization/
# with full code available at:
# https://github.com/davidsbatista/machine-learning-notebooks/blob/master/hyperparameter-across-models.ipynb
# and content licensed under https://creativecommons.org/licenses/by-nc-sa/4.0/
"""[This class performs the cross-validation for given models and parameters using GridSearchCV class]

Returns:
    [dataframe] -- [cross validation scores of models]
"""
class EstimatorSelectionHelper:

    def __init__(self, models, params):
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
        self.best_estimator = None

    def fit(self, X, y, **grid_kwargs):
        for key in self.keys:
            print('Running GridSearchCV for %s.' % key)
            model = self.models[key]
            params = self.params[key]
            grid_search = GridSearchCV(model, params, **grid_kwargs)
            grid_search.fit(X, y)
            self.grid_searches[key] = grid_search

        print('Done.')

    def score_summary(self, sort_by='mean_test_score'):
        frames = []
        for name, grid_search in self.grid_searches.items():
            frame = pd.DataFrame(grid_search.cv_results_)
            frame = frame.filter(regex='^(?!.*param_).*$')
            frame['estimator'] = len(frame) * [name]
            frames.append(frame)

        df = pd.concat(frames)

        df = df.sort_values([sort_by], ascending=False)
        df = df.reset_index()
        df = df.drop(['rank_test_score', 'index'], 1)

        columns = df.columns.tolist()
        columns.remove('estimator')
        columns = ['estimator'] + columns
        df = df[columns]
        return df

# Estimator class for Cost-Sensitive Learning using Over-Sampling with all features

class TruthfulnessCostSensitiveEstimator:

    def __init__(self):
        print('Starting Truthfulness Estimator COST-SENSITIVE Version 1.1')
        self.user_responses = self.load_survey_data()
        
        self.user_responses.drop(columns=['Prolific ID'], inplace=True)
        
        self.reordered_user_responses = self.customise_data()

        self.transformed_user_responses = self.discrete_transform()

        # This version includes only BalancedRandomForestClassifier
        # or run improved version 1.2
        # or run version 1.3 for additional classifiers
        self.classification('binary')

        # Run multi-class in version 1.2 for BalancedRandomForestClassifier
        # or version 1.3 for additional classifiers

    """[This function loads the data from csv file to pandas dataframe]
    
    Returns:
        [dataframe] -- [data]
    """
    def load_survey_data(self):
        df = pd.read_csv('All_Responses_Removed.csv')

        return df

    """[This function customises the columns]
    
    Returns:
        [dataframe] -- [data with ordered columns]
    """
    def customise_data(self):
        data = self.user_responses.copy()
        data.columns = data.columns.str.replace(r'[\s\n\t ]+', '-')
        data.columns = data.columns.str.replace(r'[a-d]-', '-')

        demographics_data = data.iloc[:, :8]
        demographics_data = demographics_data.reindex(sorted(demographics_data.columns), axis=1)
        question_data = data.reindex(sorted(data.columns[8:]), axis=1)
        reordered_user_responses = pd.concat([demographics_data, question_data], axis=1)

        return reordered_user_responses
    
    """[This function discretises age and online-presence feature data]
    
    Returns:
        [dataframe] -- [discretised data]
    """
    def discrete_transform(self):
        data = self.reordered_user_responses.copy()
        data.loc[data['Age'] <= 17, 'Age'] = 0
        data.loc[(data['Age'] > 17) & (data['Age'] <= 24), 'Age'] = 1
        data.loc[(data['Age'] > 24) & (data['Age'] <= 34), 'Age'] = 2
        data.loc[(data['Age'] > 34) & (data['Age'] <= 44), 'Age'] = 3
        data.loc[(data['Age'] > 44) & (data['Age'] <= 54), 'Age'] = 4
        data.loc[(data['Age'] > 54) & (data['Age'] <= 64), 'Age'] = 5
        data.loc[data['Age'] > 64, 'Age'] = 6

        data.loc[data['Online-Presence'] <= 5, 'Online-Presence'] = 0
        data.loc[(data['Online-Presence'] > 5) &
                (data['Online-Presence'] <= 10), 'Online-Presence'] = 1
        data.loc[(data['Online-Presence'] > 10) &
                (data['Online-Presence'] <= 15), 'Online-Presence'] = 2
        data.loc[(data['Online-Presence'] > 15) &
                (data['Online-Presence'] <= 20), 'Online-Presence'] = 3
        data.loc[(data['Online-Presence'] > 20) &
                (data['Online-Presence'] <= 25), 'Online-Presence'] = 4

        return data

    """[This function imputes missing data in any column with the mean value]
    
    Returns:
        [dataframe] -- [imputed data]
    """
    def impute_data(self, data):
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputed_data = pd.DataFrame(imp.fit_transform(
            data), columns=data.columns, index=data.index)

        return imputed_data

    """[This function transforms the data using Standard Scaler and Power Transformer object]
    
    Returns:
        [dataframe] -- [transformed training data]
    """
    def data_transformation(self, data):
        scaler = StandardScaler()
        standard_data = pd.DataFrame(scaler.fit_transform(
            data), columns=data.columns, index=data.index)

        transformer = PowerTransformer()
        transformed_data = pd.DataFrame(transformer.fit_transform(
            standard_data), columns=data.columns, index=data.index)

        return scaler, transformer, transformed_data

    """[This function calls Estimator Helper class to do cross-validation for a combination of models and parameters
    and returns the best model with its parameters]
    
    Returns:
        [dataframe] -- [best model with its parameters for the training data]
    """

    def find_clf_parameters(self, train_x, train_y, class_weights, clf_type):
        max_depth = [2, 4, 6]
        min_samples_leaf = np.arange(1, 4)
        min_samples_split = np.arange(2, 5)
        n_estimators = [100, 300]
        criterion = ['gini', 'entropy']
        sampling_strategy = ['auto', 'majority', 'not majority']
        class_weight_param = [class_weights]

        models1 = {
            'BalancedRandomForestClassifier': BalancedRandomForestClassifier(random_state=42)
        }

        params1 = {
            'BalancedRandomForestClassifier': [
                {'criterion': criterion, 'n_estimators': n_estimators, 'max_depth': max_depth, 
                 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split,
                 'sampling_strategy': sampling_strategy, 'class_weight': class_weight_param
                }]
        }

        helper1 = EstimatorSelectionHelper(models1, params1)
        if clf_type == 'binary':
            helper1.fit(train_x, train_y, cv=5, scoring='balanced_accuracy', n_jobs=-1)
        df = helper1.score_summary()
        best_estimator = df['estimator'].iloc[0]
        best_estimator_params = df['params'].iloc[0]

        return best_estimator, best_estimator_params

    """[This function splits the data and makes a call to find the best model for the training data]
    """
    def classification(self, clf_type):
        class_weights = {}
        data = self.transformed_user_responses.copy()
        if clf_type == 'binary':
            data.iloc[:, 10:207:4] = (data.iloc[:, 10:207:4] == 7.0)

        question_estimator = {}

        relevant_indexes = []
        demographics_column_indexes = ['Age', 'Gender', 'IUIPC-Awareness', 'IUIPC-Collection', 'IUIPC-Control',
                                    'Online-Presence', 'Personal-Stability', 'Reciprocity']
        relevant_indexes.extend(demographics_column_indexes)

        for question_number in range(1, 51):
            print("Question No. is: ", question_number)
            question = 'Q' + str(question_number).zfill(2)
            
            question_indexes = []
            question_indexes.extend([str(question_number).zfill(2) + '-Effort',
                            str(question_number).zfill(2) + '-Relevance',
                            str(question_number).zfill(2) + '-Uncomfortable',
                            str(question_number).zfill(2) + '-Truthfulness'])
            relevant_indexes.extend(question_indexes)

            question_label = str(question_number).zfill(2) + '-Truthfulness'

            if clf_type == 'binary':
                train_data_question, test_data_question = train_test_split(data[relevant_indexes], 
                stratify=data[question_label], test_size=0.3, random_state=42)

            train_x_question = train_data_question.copy()
            if clf_type == 'binary':
                train_x_question = self.impute_data(train_x_question)

                computed_weights = class_weight.compute_class_weight('balanced', np.unique(train_x_question[question_label]), train_x_question[question_label]).tolist()
                class_weights[question] = {0:computed_weights[0], 1:computed_weights[1]}
            
            train_y_question = train_x_question.loc[:, question_label]
            train_x_question.drop(columns=question_label, inplace=True)

            train_scaler_question, train_transformer_question, transformed_train_x_question = \
                self.data_transformation(train_x_question)

            best_estimator, best_estimator_params = self.find_clf_parameters(transformed_train_x_question, 
            train_y_question, class_weights[question], clf_type)
            question_estimator[question] = [best_estimator, best_estimator_params]

            del relevant_indexes[8:]

        if not os.path.isdir(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)

        filename = os.path.join(
            RESULTS_DIR, clf_type + '_cost_sensitive_estimator_parameters_1.1.txt')

        with open(filename, 'w') as f:
            print(question_estimator, file=f)    


cost_sensitive_estimator = TruthfulnessCostSensitiveEstimator()