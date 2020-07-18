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

from sklearn.ensemble import RandomForestClassifier

from imblearn.ensemble import BalancedRandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from sklearn.utils import class_weight

import warnings

warnings.filterwarnings('ignore')
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

RESULTS_DIR = 'results'

# Predictor class for Cost-Sensitive Learning with informative features


class TruthfulnessCostSensitivePredictor:

    def __init__(self):
        print("Starting Truthfulness Predictor COST-SENSITIVE Version 1.2")
        self.user_responses = self.load_survey_data()

        self.user_responses.drop(columns=['Prolific ID'], inplace=True)

        self.reordered_user_responses = self.customise_data()

        self.transformed_user_responses = self.discrete_transform()

        # This version includes only BalancedRandomForestClassifier
        # Run version 1.3 for additional classifiers
        self.classification('binary')
        # Run for multi-class in version 1.3

    def __final__(self):
        print("Ending Truthfulness Predictor COST-SENSITIVE Version 1.2")

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

        data = self.user_responses
        data.columns = data.columns.str.replace(r'[\s\n\t ]+', '-')
        data.columns = data.columns.str.replace(r'[a-d]-', '-')

        demographics_data = data.iloc[:, :8]
        demographics_data = demographics_data.reindex(
            sorted(demographics_data.columns), axis=1)
        question_data = data.reindex(sorted(data.columns[8:]), axis=1)
        reordered_user_responses = pd.concat(
            [demographics_data, question_data], axis=1)

        return reordered_user_responses

    """[This function discretises age and online-presence feature data]
    
    Returns:
        [dataframe] -- [discretised data]
    """
    def discrete_transform(self):

        data = self.reordered_user_responses
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
    def train_data_transformation(self, data):
        scaler = StandardScaler()
        standard_data = pd.DataFrame(scaler.fit_transform(
            data), columns=data.columns, index=data.index)

        transformer = PowerTransformer()
        transformed_data = pd.DataFrame(transformer.fit_transform(
            standard_data), columns=data.columns, index=data.index)

        return scaler, transformer, transformed_data

    """[This function applies the transformation on the test data]
    
    Returns:
        [dataframe] -- [transformed test data]
    """
    def test_data_transformation(self, train_scaler, train_transformer, data):
        standard_data = pd.DataFrame(train_scaler.fit_transform(
            data), columns=data.columns, index=data.index)

        transformed_data = pd.DataFrame(train_transformer.fit_transform(
            standard_data), columns=data.columns, index=data.index)

        return transformed_data

    """[This function transforms the data back to original form]
    
    Returns:
        [dataframe] -- [original data]
    """
    def data_inverse_transformation(self, scaler_object, transformer_object, data):
        inverse_transformed_data = pd.DataFrame(transformer_object.inverse_transform(data),
                                                columns=data.columns, index=data.index)
        inverse_scaled_data = pd.DataFrame(scaler_object.inverse_transform(inverse_transformed_data),
                                           columns=data.columns, index=data.index)

        return inverse_scaled_data

    """[This function initialises the naive binary classifiers]
    
    Returns:
        [list] -- [instances of naive binary classifiers]
    """
    def get_naive_estimator(self, class_weights):
        balanced_rf_clf = BalancedRandomForestClassifier(
            sampling_strategy='not majority', class_weight=class_weights, random_state=42)

        naive_estimator = [balanced_rf_clf]

        return naive_estimator

    """[This function calculates the cross-validation scores for a question]
    
    Returns:
        [dictionary] -- [cross-val scores]
    """
    def calc_cross_val_scores(self, clf, data, labels, clf_type, question):
        scores = {}

        predicted_scores = cross_val_score(
            clf, data, labels, cv=10, scoring="accuracy")
        predicted_labels = cross_val_predict(clf, data, labels, cv=10)

        if clf_type == 'binary':
            tn, fp, fn, tp = confusion_matrix(labels, predicted_labels).ravel()
            specificity = round((tn / (tn + fp)) * 100, 2)

        predicted_prob = clf.predict_proba(data)
        predicted_prob_true = [p[1] for p in predicted_prob]

        scores['Question'] = question
        scores['Accuracy'] = round(predicted_scores.mean() * 100, 2)
        scores['Balanced Accuracy'] = round(
            balanced_accuracy_score(labels, predicted_labels) * 100, 2)
        if clf_type == 'binary':
            scores['Precision'] = round(precision_score(
                labels, predicted_labels) * 100, 2)
            scores['Recall'] = round(recall_score(
                labels, predicted_labels) * 100, 2)
            scores['Specificity'] = specificity
            scores['F1'] = round(f1_score(labels, predicted_labels), 2)
            scores['ROC AUC'] = round(
                roc_auc_score(labels, predicted_prob_true), 2)

        if clf_type == 'multi-class':
            scores['Precision'] = round(precision_score(
                labels, predicted_labels, average='micro'), 2)
            scores['Recall'] = round(recall_score(
                labels, predicted_labels, average='micro'), 2)
            scores['Specificity'] = 'NA'
            scores['F1'] = round(
                f1_score(labels, predicted_labels, average='micro'), 2)
            scores['ROC AUC'] = 'NA'

        # print('Confusion Matrix for Q-%s is: ' % (str(question).zfill(2)))
        # print(confusion_matrix(labels, predicted_labels))

        return scores

    """[This function trains the naive classifiers]
    
    Returns:
        [list] -- [cross validation scores of all classifiers for a question]
    """
    def run_naive_classifiers(self, train_x, train_y, class_weights, clf_type, question):
        naive_estimator = self.get_naive_estimator(class_weights)
        clf_scores = []

        for estimator in range(len(naive_estimator)):
            clf = naive_estimator[estimator]
            clf.fit(train_x, train_y)
            scores = self.calc_cross_val_scores(
                clf, train_x, train_y, clf_type, question)
            clf_scores.append(scores)

        # print('Perfect Confusion Matrix for Q-%s is: ' % (str(question).zfill(2)))
        # perfect_labels = train_y
        # print(confusion_matrix(train_y, perfect_labels))

        return scores

    """[This function creates a dataframe to save scores for naive classifiers]
    
    Returns:
        [dataframe] -- [score dataframe for naive classifiers]
    """
    def create_score_frame(self, scores):

        balanced_rf_clf_scores = scores

        clf_score_frame = pd.DataFrame({
            'Question': [balanced_rf_clf_scores['Question']],
            'Model': ['BalancedRandomForestClassifier'],
            'Accuracy': [balanced_rf_clf_scores['Accuracy']],
            'Balanced Accuracy': [balanced_rf_clf_scores['Balanced Accuracy']],
            'Precision': [balanced_rf_clf_scores['Precision']],
            'Recall': [balanced_rf_clf_scores['Recall']],
            'Specificity': [balanced_rf_clf_scores['Specificity']],
            'F1': [balanced_rf_clf_scores['F1']],
            'ROC AUC': [balanced_rf_clf_scores['ROC AUC']]
        }, columns=['Question', 'Model', 'Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 'ROC AUC'])

        return clf_score_frame

    """[This function creates a dataframe to save scores for the best classifier ]
    
    Returns:
        [dataframe] -- [score dataframe for the best classifier]
    """
    def create_score_frame_best_estimator(self, estimator, scores):
        clf_score_frame = pd.DataFrame({
            'Question': scores['Question'],
            'Model': estimator,
            'Accuracy': scores['Accuracy'],
            'Balanced Accuracy': scores['Balanced Accuracy'],
            'Precision': scores['Precision'],
            'Recall': scores['Recall'],
            'Specificity': scores['Specificity'],
            'F1': scores['F1'],
            'ROC AUC': scores['ROC AUC']
        }, index=['Question'],
            columns=['Question', 'Model', 'Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 'ROC AUC'])

        return clf_score_frame

    """[This function makes a call to run naive binary classifiers]
    
    Returns:
        [dataframe] -- [score dataframe for naive binary classifiers]
    """
    def naive_classification(self, train_x, train_y, class_weights, clf_type, question):
        scores = self.run_naive_classifiers(
            train_x, train_y, class_weights, clf_type, question)
        clf_score_frame = self.create_score_frame(scores)

        return clf_score_frame

    """[This function provides the best binary classifiers manually retrieved from Estimator Finder component for all 50 questions]
    
    Returns:
        [dictionary] -- [instances of best binary classifiers]
    """
    def get_binary_estimators(self):
        # cv=5 balanced_accuracy
        binary_estimators = {'Q01': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 3.480286738351255}, 'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 2, 'min_samples_split': 3, 'n_estimators': 100, 'sampling_strategy': 'not majority'}], 'Q02': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 1.4807740014884643}, 'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q03': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 2.2467532467532467}, 'criterion': 'entropy', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 100, 'sampling_strategy': 'not majority'}], 'Q04': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 9.319917440660475}, 'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q05': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 1.9949086552860138}, 'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q06': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 2.091190108191654}, 'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q07': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 3.3140638481449525}, 'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q08': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 5.963788300835655}, 'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 100, 'sampling_strategy': 'majority'}], 'Q09': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 3.0144520272982738}, 'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q10': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 3.0420371867421183}, 'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 3, 'min_samples_split': 3, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q11': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 4.611672278338945}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 300, 'sampling_strategy': 'auto'}], 'Q12': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 10.11111111111111}, 'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'majority'}], 'Q13': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 5.020469596628537}, 'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q14': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 5.28140703517588}, 'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q15': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 6.412898443291327}, 'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q16': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 1.501876407305479}, 'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q17': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 2.3602150537634405}, 'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 100, 'sampling_strategy': 'majority'}], 'Q18': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 1.7525461051472613}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 100, 'sampling_strategy': 'majority'}], 'Q19': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 1.7262813522355507}, 'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 300, 'sampling_strategy': 'majority'}], 'Q20': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 1.9638411381149972}, 'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 3, 'min_samples_split': 4, 'n_estimators': 100, 'sampling_strategy': 'majority'}], 'Q21': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 2.3025099075297226}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 3, 'min_samples_split': 4, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q22': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 2.7537537537537538}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 300, 'sampling_strategy': 'auto'}], 'Q23': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 1.9797377830750893}, 'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 100, 'sampling_strategy': 'not majority'}], 'Q24': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 2.3025099075297226}, 'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'not majority'}], 'Q25': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 1.7785495971103085}, 'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q26': [
            'BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 4.302226935312832}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 3, 'min_samples_split': 4, 'n_estimators': 100, 'sampling_strategy': 'not majority'}], 'Q27': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 2.158559696778269}, 'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 3, 'min_samples_split': 4, 'n_estimators': 300, 'sampling_strategy': 'majority'}], 'Q28': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 1.6759432700026762}, 'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q29': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 2.265839320705421}, 'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 100, 'sampling_strategy': 'majority'}], 'Q30': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 3.897159647404505}, 'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 3, 'min_samples_split': 3, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q31': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 3.128819157720892}, 'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q32': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 2.284072249589491}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q33': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 1.9342723004694837}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 3, 'min_samples_split': 4, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q34': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 2.8789759503491075}, 'criterion': 'entropy', 'max_depth': 2, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'not majority'}], 'Q35': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 2.8535645472061657}, 'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 2, 'min_samples_split': 3, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q36': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 2.5906642728904847}, 'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 300, 'sampling_strategy': 'majority'}], 'Q37': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 2.175611305176247}, 'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q38': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 1.8058361391694724}, 'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 3, 'min_samples_split': 3, 'n_estimators': 300, 'sampling_strategy': 'majority'}], 'Q39': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 4.025125628140704}, 'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q40': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 2.932363350373574}, 'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q41': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 2.2467532467532467}, 'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q42': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 2.4199726402188784}, 'criterion': 'entropy', 'max_depth': 2, 'min_samples_leaf': 3, 'min_samples_split': 4, 'n_estimators': 300, 'sampling_strategy': 'auto'}], 'Q43': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 3.0144520272982738}, 'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 100, 'sampling_strategy': 'majority'}], 'Q44': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 1.8752156411730878}, 'criterion': 'entropy', 'max_depth': 2, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'not majority'}], 'Q45': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 2.7537537537537538}, 'criterion': 'entropy', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q46': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 4.302226935312832}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 300, 'sampling_strategy': 'majority'}], 'Q47': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 1.918855808523059}, 'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q48': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 1.8612303290414876}, 'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300, 'sampling_strategy': 'majority'}], 'Q49': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 2.5906642728904847}, 'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300, 'sampling_strategy': 'majority'}], 'Q50': ['BalancedRandomForestClassifier', {'class_weight': {False: 1, True: 13.104372355430183}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 100, 'sampling_strategy': 'majority'}]}

        return binary_estimators

    """[This function provides the best multiclass models manually retrieved from Estimator Finder component for all 50 questions]
    
    Returns:
        [dictionary] -- [instances of best binary classifiers]
    """
    def get_multi_class_estimators(self):

        multi_class_estimators = {'Q01': ['BalancedRandomForestClassifier', {'class_weight': {0: 1.4183006535947713, 1: 1.954954954954955, 2: 0.5607235142118863}, 'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q02': ['BalancedRandomForestClassifier', {'class_weight': {0: 27.0, 1: 11.571428571428571, 2: 0.34763948497854075}, 'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300, 'sampling_strategy': 'majority'}], 'Q03': [
            'BalancedRandomForestClassifier', {'class_weight': {0: 2.212962962962963, 1: 3.0641025641025643, 2: 0.4500941619585687}, 'criterion': 'entropy', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 100, 'sampling_strategy': 'not majority'}], 'Q04': ['BalancedRandomForestClassifier', {'class_weight': {0: 0.4927536231884058, 1: 3.7777777777777777, 2: 1.4166666666666667}, 'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 300, 'sampling_strategy': 'majority'}]}

        return multi_class_estimators

    """[This function trains and evaluates the best classifiers for all questions]
    
    Returns:
        [dictionaries] -- [cross-validation and prediction scores]
    """
    def run_best_estimator(self, train_x, train_y, test_x, test_y, estimator, params, clf_type, question):
        estimator_scores = {}

        if estimator == 'BalancedRandomForestClassifier':
            clf = BalancedRandomForestClassifier(n_estimators=params['n_estimators'], criterion=params['criterion'],
                                                 max_depth=params['max_depth'], sampling_strategy=params['sampling_strategy'],
                                                 min_samples_leaf=params['min_samples_leaf'], min_samples_split=params['min_samples_split'],
                                                 class_weight=params['class_weight'], random_state=42)

        clf.fit(train_x, train_y)
        cross_val_scores = self.calc_cross_val_scores(
            clf, train_x, train_y, clf_type, question)

        predicted_labels = clf.predict(test_x)

        if clf_type == 'binary':
            tn, fp, fn, tp = confusion_matrix(test_y, predicted_labels).ravel()
            specificity = round((tn / (tn + fp)) * 100, 2)

        predicted_prob = clf.predict_proba(test_x)
        predicted_prob_true = [p[1] for p in predicted_prob]
        estimator_scores['Question'] = question
        estimator_scores['Accuracy'] = round(
            accuracy_score(test_y, predicted_labels) * 100, 2)
        estimator_scores['Balanced Accuracy'] = round(
            balanced_accuracy_score(test_y, predicted_labels) * 100, 2)
        if clf_type == 'binary':
            estimator_scores['Precision'] = round(
                precision_score(test_y, predicted_labels) * 100, 2)
            estimator_scores['Recall'] = round(
                recall_score(test_y, predicted_labels) * 100, 2)
            estimator_scores['Specificity'] = specificity
            estimator_scores['F1'] = round(
                f1_score(test_y, predicted_labels), 2)
            estimator_scores['ROC AUC'] = round(
                roc_auc_score(test_y, predicted_prob_true), 2)

        if clf_type == 'multi-class':
            estimator_scores['Precision'] = round(
                precision_score(test_y, predicted_labels, average='micro') * 100, 2)
            estimator_scores['Recall'] = round(
                recall_score(test_y, predicted_labels, average='micro') * 100, 2)
            estimator_scores['Specificity'] = 'NA'
            estimator_scores['F1'] = round(
                f1_score(test_y, predicted_labels, average='micro'), 2)
            estimator_scores['ROC AUC'] = 'NA'

        # print('Perfect Confusion Matrix for Q-%s is: ' % (str(question).zfill(2)))
        # perfect_labels = train_y
        # print(confusion_matrix(train_y, perfect_labels))

        return cross_val_scores, estimator_scores
        
    """[This function performs classification]
    """
    def classification(self, clf_type):

        class_weights = {}
        important_features = {}
        relevant_indexes = []
        demographics_column_indexes = ['Age', 'Gender', 'IUIPC-Awareness', 'IUIPC-Collection', 'IUIPC-Control',
                                       'Online-Presence', 'Personal-Stability', 'Reciprocity']
        relevant_indexes.extend(demographics_column_indexes)

        naive_clf_scores = pd.DataFrame(columns=[
                                        'Question', 'Model', 'Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 'ROC AUC'])
        cross_val_scores = pd.DataFrame(columns=[
                                        'Question', 'Model', 'Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 'ROC AUC'])
        best_clf_scores = pd.DataFrame(
            columns=['Question', 'Model', 'Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 'ROC AUC'])

        data = self.transformed_user_responses
        if clf_type == 'binary':
            data.iloc[:, 10:207:4] = (data.iloc[:, 10:207:4] == 7.0)
        if clf_type == 'multi-class':
            data.iloc[:, 10:207:4] = data.iloc[:, 10:207:4].replace(
                [1.0, 2.0, 3.0], np.float64(0))
            data.iloc[:, 10:207:4] = data.iloc[:, 10:207:4].replace(
                [4.0, 5.0, 6.0], np.float64(1))
            data.iloc[:, 10:207:4] = data.iloc[:,
                                               10:207:4].replace([7.0], np.float64(2))

        for question_number in range(1, 51):
            question = 'Q' + str(question_number).zfill(2)
            question_indexes = []
            question_indexes.extend([str(question_number).zfill(2) + '-Effort',
                                     str(question_number).zfill(
                                         2) + '-Relevance',
                                     str(question_number).zfill(
                                         2) + '-Uncomfortable',
                                     str(question_number).zfill(2) + '-Truthfulness'])
            relevant_indexes.extend(question_indexes)

            question_label = str(question_number).zfill(2) + '-Truthfulness'

            if clf_type == 'binary':
                train_data_question, test_data_question = train_test_split(
                    data[relevant_indexes], stratify=data[question_label], test_size=0.3,
                    random_state=42)
            if clf_type == 'multi-class':
                cleaned_user_responses = data.loc[:, relevant_indexes].dropna()

                train_data_question, test_data_question = train_test_split(cleaned_user_responses,
                                                                           stratify=cleaned_user_responses[question_label],
                                                                           test_size=0.3, random_state=42)

            train_x_question = train_data_question.copy()
            test_x_question = test_data_question.copy()

            if clf_type == 'binary':
                train_x_question = self.impute_data(train_x_question)
                test_x_question = self.impute_data(test_x_question)

                computed_weights = class_weight.compute_class_weight('balanced', np.unique(
                    train_x_question[question_label]), train_x_question[question_label]).tolist()
                class_weights[question] = {
                    0: computed_weights[0], 1: computed_weights[1]}

            if clf_type == 'multi-class':
                computed_weights = class_weight.compute_class_weight('balanced', np.unique(
                    train_x_question[question_label]), train_x_question[question_label]).tolist()
                class_weights[question] = {
                    0: computed_weights[0], 1: computed_weights[1], 2: computed_weights[2]}

            train_y_question = train_x_question.loc[:, question_label]
            test_y_question = test_x_question.loc[:, question_label]
            train_x_question.drop(columns=question_label, inplace=True)
            test_x_question.drop(columns=question_label, inplace=True)

            train_scaler_question, train_transformer_question, transformed_train_x_question = \
                self.train_data_transformation(train_x_question)

            transformed_test_x_question = \
                self.test_data_transformation(
                    train_scaler_question, train_transformer_question, test_x_question)

            # Feature Selection
            rf = RandomForestClassifier(n_estimators=300, random_state=42)
            rf.fit(transformed_train_x_question, train_y_question)

            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1]

            important_features[question] = indices[0:3].tolist()

            featured_train_data = transformed_train_x_question.iloc[:,
                                                                    important_features[question]]
            featured_test_data = transformed_test_x_question.iloc[:,
                                                                  important_features[question]]

            print(clf_type + ' Classification for ' + question + ' Started.')
            clf_score_frame = self.naive_classification(featured_train_data, train_y_question,
                                                        class_weights[question], clf_type, question)
            naive_clf_scores = naive_clf_scores.append(clf_score_frame)

            if clf_type == 'binary':
                binary_estimators = self.get_binary_estimators()
                validated_estimator = binary_estimators[question]
            if clf_type == 'multi-class':
                multi_class_estimators = self.get_multi_class_estimators()
                validated_estimator = multi_class_estimators[question]

            best_estimator = validated_estimator[0]
            best_estimator_params = validated_estimator[1]

            best_clf_cross_val_scores, best_estimator_scores = self.run_best_estimator(featured_train_data, train_y_question,
                                                                                       featured_test_data, test_y_question,
                                                                                       best_estimator, best_estimator_params, clf_type, question)
            print(clf_type + ' Classification for ' + question + ' Completed.')

            cross_val_frame = self.create_score_frame_best_estimator(
                best_estimator, best_clf_cross_val_scores)

            cross_val_scores = cross_val_scores.append(cross_val_frame)

            best_clf_frame = self.create_score_frame_best_estimator(
                best_estimator, best_estimator_scores)

            best_clf_scores = best_clf_scores.append(best_clf_frame)

            del relevant_indexes[8:]

        naive_clf_scores = naive_clf_scores.set_index(['Question'])
        cross_val_scores = cross_val_scores.set_index(['Question'])
        best_clf_scores = best_clf_scores.set_index(['Question'])

        if clf_type == 'multi-class':
            naive_clf_scores = naive_clf_scores.drop(
                columns=['Specificity', 'ROC AUC'])
            cross_val_scores = cross_val_scores.drop(
                columns=['Specificity', 'ROC AUC'])
            best_clf_scores = best_clf_scores.drop(
                columns=['Specificity', 'ROC AUC'])

        if not os.path.isdir(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)

        naive_clf_file_path = os.path.join(
            RESULTS_DIR, clf_type + '_naive_cost_sensitive_clf_cross_val_scores_1.2.xlsx')
        best_cross_val_file_path = os.path.join(
            RESULTS_DIR, clf_type + '_best_clf_cost_sensitive_cross_val_scores_1.2.xlsx')
        best_clf_file_path = os.path.join(
            RESULTS_DIR, clf_type + '_best_clf_cost_sensitive_accuracy_scores_1.2.xlsx')
        # Write results to the EXCEL files
        naive_clf_scores.to_excel(naive_clf_file_path)
        cross_val_scores.to_excel(best_cross_val_file_path)
        best_clf_scores.to_excel(best_clf_file_path)


predictor = TruthfulnessCostSensitivePredictor()
