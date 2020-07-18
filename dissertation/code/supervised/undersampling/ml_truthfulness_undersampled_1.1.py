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

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import EasyEnsembleClassifier

from imblearn.over_sampling import SMOTE

from sklearn.tree import export_graphviz

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, balanced_accuracy_score

import warnings

warnings.filterwarnings('ignore')

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

BINARY_RESULTS_DIR = 'results'

# Predictor class for Imbalanced Learning using Undersampling with all features
class TruthfulnessUndersampledPredictor:

    def __init__(self):
        print('Starting Truthfulness Predictor UNDERSAMPLED Version 1.1')
        self.user_responses = self.load_survey_data()

        self.user_responses.drop(columns=['Prolific ID'], inplace=True)

        self.reordered_user_responses = self.customise_data()

        self.transformed_user_responses = self.discrete_transform()

        self.classification('binary')
        # Run for multi-class in version 1.2

    def __final__(self):
        print('Ending Truthfulness Predictor UNDERSAMPLED Version 1.1')

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

    def get_naive_binary_estimators(self):
        balanced_rf_clf = BalancedRandomForestClassifier(
            sampling_strategy='not majority', random_state=42)
        balanced_bagging_clf = BalancedBaggingClassifier(
            sampling_strategy='not majority', random_state=42)
        balanced_ensemble_clf = EasyEnsembleClassifier(
            sampling_strategy='not majority', random_state=42)

        binary_estimators = [balanced_rf_clf,
                             balanced_bagging_clf, balanced_ensemble_clf]

        return binary_estimators

    """[This function calculates the cross-validation scores for a question]
    
    Returns:
        [dictionary] -- [cross-val scores]
    """
    def calc_cross_val_scores(self, clf, data, labels, clf_type, question):
        scores = {}

        predicted_scores = cross_val_score(
            clf, data, labels, cv=10, scoring="accuracy")
        predicted_labels = cross_val_predict(clf, data, labels, cv=10)

        tn, fp, fn, tp = confusion_matrix(labels, predicted_labels).ravel()
        specificity = round((tn / (tn + fp)) * 100, 2)

        predicted_prob = clf.predict_proba(data)
        predicted_prob_true = [p[1] for p in predicted_prob]
        scores['Question'] = question
        scores['Accuracy'] = round(predicted_scores.mean() * 100, 2)
        scores['Balanced Accuracy'] = round(
            balanced_accuracy_score(labels, predicted_labels) * 100, 2)
        scores['Precision'] = round(precision_score(
            labels, predicted_labels) * 100, 2)
        scores['Recall'] = round(recall_score(
            labels, predicted_labels) * 100, 2)
        scores['Specificity'] = specificity
        scores['F1'] = round(f1_score(labels, predicted_labels), 2)
        scores['ROC AUC'] = round(
            roc_auc_score(labels, predicted_prob_true), 2)

        # print('Confusion Matrix for Q-%s is: ' % (str(question).zfill(2)))
        # print(confusion_matrix(labels, predicted_labels))

        return scores

    """[This function trains the naive binary classifiers]
    
    Returns:
        [list] -- [cross validation scores of all classifiers for a question]
    """
    def run_naive_binary_classifiers(self, train_x, train_y, clf_type, question):
        naive_binary_estimators = self.get_naive_binary_estimators()
        clf_scores = []

        for estimator in range(len(naive_binary_estimators)):
            clf = naive_binary_estimators[estimator]
            clf.fit(train_x, train_y)
            scores = self.calc_cross_val_scores(
                clf, train_x, train_y, clf_type, question)
            clf_scores.append(scores)

        # print('Perfect Confusion Matrix for Q-%s is: ' % (str(question).zfill(2)))
        # perfect_labels = train_y
        # print(confusion_matrix(train_y, perfect_labels))

        return clf_scores

    """[This function creates a dataframe to save scores for naive binary classifiers]
    
    Returns:
        [dataframe] -- [score dataframe for naive binary classifiers]
    """
    def create_score_frame(self, scores):
        balanced_rf_clf_scores, balanced_bagging_clf_scores, balanced_ensemble_clf_scores = \
            [scores[i] for i in range(len(scores))]

        clf_score_frame = pd.DataFrame({
            'Question': [balanced_rf_clf_scores['Question'], balanced_bagging_clf_scores['Question'],
                         balanced_ensemble_clf_scores['Question']],
            'Model': ['Balanced Random Forest', 'Balanced Bagging', 'Easy Ensemble'],
            'Accuracy': [balanced_rf_clf_scores['Accuracy'], balanced_bagging_clf_scores['Accuracy'],
                         balanced_ensemble_clf_scores['Accuracy']],
            'Balanced Accuracy': [balanced_rf_clf_scores['Balanced Accuracy'], balanced_bagging_clf_scores['Balanced Accuracy'],
                                  balanced_ensemble_clf_scores['Balanced Accuracy']],
            'Precision': [balanced_rf_clf_scores['Precision'], balanced_bagging_clf_scores['Precision'],
                          balanced_ensemble_clf_scores['Precision']],
            'Recall': [balanced_rf_clf_scores['Recall'], balanced_bagging_clf_scores['Recall'],
                       balanced_ensemble_clf_scores['Recall']],
            'Specificity': [balanced_rf_clf_scores['Specificity'], balanced_bagging_clf_scores['Specificity'],
                            balanced_ensemble_clf_scores['Specificity']],
            'F1': [balanced_rf_clf_scores['F1'], balanced_bagging_clf_scores['F1'], balanced_ensemble_clf_scores['F1']],
            'ROC AUC': [balanced_rf_clf_scores['ROC AUC'], balanced_bagging_clf_scores['ROC AUC'],
                        balanced_ensemble_clf_scores['ROC AUC']]
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
    def naive_binary_classification(self, train_x, train_y, clf_type, question):
        scores = self.run_naive_binary_classifiers(
            train_x, train_y, clf_type, question)
        clf_score_frame = self.create_score_frame(scores)

        return clf_score_frame

    """[This function provides the best binary classifiers manually retrieved from Estimator Finder component for all 50 questions]
    
    Returns:
        [dictionary] -- [instances of best binary classifiers]
    """
    def get_binary_estimators(self):
        #  cv = 5 balanced_accuracy
        binary_estimators = {'Q01': ['BalancedRandomForestClassifier', {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 100, 'sampling_strategy': 'not majority'}], 'Q02': ['BalancedRandomForestClassifier', {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q03': ['BalancedRandomForestClassifier', {'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q04': ['EasyEnsembleClassifier', {'n_estimators': 100, 'sampling_strategy': 'majority'}], 'Q05': ['BalancedRandomForestClassifier', {'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q06': ['BalancedRandomForestClassifier', {'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 3, 'min_samples_split': 4, 'n_estimators': 100, 'sampling_strategy': 'not majority'}], 'Q07': ['BalancedRandomForestClassifier', {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 300, 'sampling_strategy': 'auto'}], 'Q08': ['EasyEnsembleClassifier', {'n_estimators': 300, 'sampling_strategy': 'majority'}], 'Q09': ['BalancedRandomForestClassifier', {'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 300, 'sampling_strategy': 'majority'}], 'Q10': ['BalancedRandomForestClassifier', {'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 100, 'sampling_strategy': 'majority'}], 'Q11': ['EasyEnsembleClassifier', {'n_estimators': 300, 'sampling_strategy': 'majority'}], 'Q12': ['EasyEnsembleClassifier', {'n_estimators': 100, 'sampling_strategy': 'majority'}], 'Q13': ['BalancedRandomForestClassifier', {'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 2, 'min_samples_split': 3, 'n_estimators': 300, 'sampling_strategy': 'majority'}], 'Q14': ['BalancedRandomForestClassifier', {'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'majority'}], 'Q15': ['BalancedRandomForestClassifier', {'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q16': ['BalancedRandomForestClassifier', {'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q17': ['BalancedRandomForestClassifier', {'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'majority'}], 'Q18': ['BalancedRandomForestClassifier', {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q19': ['BalancedRandomForestClassifier', {'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'majority'}], 'Q20': ['BalancedRandomForestClassifier', {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q21': ['BalancedRandomForestClassifier', {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 300, 'sampling_strategy': 'majority'}], 'Q22': ['BalancedRandomForestClassifier', {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 300, 'sampling_strategy': 'majority'}], 'Q23': ['BalancedRandomForestClassifier', {'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 2, 'min_samples_split': 3, 'n_estimators': 100, 'sampling_strategy': 'majority'}], 'Q24': ['EasyEnsembleClassifier', {'n_estimators': 100, 'sampling_strategy': 'majority'}], 'Q25': ['BalancedRandomForestClassifier', {'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q26': ['BalancedRandomForestClassifier', {'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q27': [
            'BalancedRandomForestClassifier', {'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q28': ['BalancedRandomForestClassifier', {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q29': ['BalancedRandomForestClassifier', {'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300, 'sampling_strategy': 'majority'}], 'Q30': ['BalancedRandomForestClassifier', {'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q31': ['BalancedRandomForestClassifier', {'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q32': ['BalancedRandomForestClassifier', {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300, 'sampling_strategy': 'majority'}], 'Q33': ['BalancedRandomForestClassifier', {'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'not majority'}], 'Q34': ['BalancedRandomForestClassifier', {'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q35': ['BalancedRandomForestClassifier', {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 300, 'sampling_strategy': 'majority'}], 'Q36': ['BalancedRandomForestClassifier', {'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 300, 'sampling_strategy': 'majority'}], 'Q37': ['BalancedRandomForestClassifier', {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q38': ['BalancedRandomForestClassifier', {'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 100, 'sampling_strategy': 'majority'}], 'Q39': ['BalancedRandomForestClassifier', {'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 3, 'min_samples_split': 4, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q40': ['BalancedRandomForestClassifier', {'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 100, 'sampling_strategy': 'majority'}], 'Q41': ['BalancedRandomForestClassifier', {'criterion': 'entropy', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300, 'sampling_strategy': 'auto'}], 'Q42': ['BalancedRandomForestClassifier', {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q43': ['BalancedRandomForestClassifier', {'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q44': ['BalancedRandomForestClassifier', {'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 2, 'min_samples_split': 3, 'n_estimators': 100, 'sampling_strategy': 'majority'}], 'Q45': ['BalancedRandomForestClassifier', {'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q46': ['BalancedRandomForestClassifier', {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'majority'}], 'Q47': ['BalancedRandomForestClassifier', {'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'majority'}], 'Q48': ['EasyEnsembleClassifier', {'n_estimators': 300, 'sampling_strategy': 'majority'}], 'Q49': ['BalancedRandomForestClassifier', {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'majority'}], 'Q50': ['BalancedRandomForestClassifier', {'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 300, 'sampling_strategy': 'auto'}]}

        return binary_estimators
    
    """[This function trains and evaluates the best classifiers for all questions]
    
    Returns:
        [dictionaries] -- [cross-validation and prediction scores]
    """
    def run_best_estimator(self, train_x, train_y, test_x, test_y, estimator, params, clf_type, question):
        estimator_scores = {}

        if estimator == 'BalancedRandomForestClassifier':
            clf = BalancedRandomForestClassifier(n_estimators=params['n_estimators'],
                                                 sampling_strategy=params['sampling_strategy'],
                                                 random_state=42)
        elif estimator == 'BalancedBaggingClassifier':
            clf = BalancedBaggingClassifier(n_estimators=params['n_estimators'], bootstrap=params['bootstrap'],
                                            max_samples=params['max_samples'], sampling_strategy=params['sampling_strategy'],
                                            random_state=42)
        elif estimator == 'EasyEnsembleClassifier':
            clf = EasyEnsembleClassifier(n_estimators=params['n_estimators'],
                                         sampling_strategy=params['sampling_strategy'],
                                         random_state=42)

        clf.fit(train_x, train_y)
        cross_val_scores = self.calc_cross_val_scores(
            clf, train_x, train_y, clf_type, question)

        predicted_labels = clf.predict(test_x)

        tn, fp, fn, tp = confusion_matrix(test_y, predicted_labels).ravel()
        specificity = round((tn / (tn + fp)) * 100, 2)

        predicted_prob = clf.predict_proba(test_x)
        predicted_prob_true = [p[1] for p in predicted_prob]

        estimator_scores['Question'] = question
        estimator_scores['Accuracy'] = round(
            accuracy_score(test_y, predicted_labels) * 100, 2)
        estimator_scores['Balanced Accuracy'] = round(
            balanced_accuracy_score(test_y, predicted_labels) * 100, 2)
        estimator_scores['Precision'] = round(
            precision_score(test_y, predicted_labels) * 100, 2)
        estimator_scores['Recall'] = round(
            recall_score(test_y, predicted_labels) * 100, 2)
        estimator_scores['Specificity'] = specificity
        estimator_scores['F1'] = round(
            f1_score(test_y, predicted_labels), 2)
        estimator_scores['ROC AUC'] = round(
            roc_auc_score(test_y, predicted_prob_true), 2)

        # print('Perfect Confusion Matrix for Q-%s is: ' % (str(question).zfill(2)))
        # perfect_labels = train_y
        # print(confusion_matrix(train_y, perfect_labels))

        return cross_val_scores, estimator_scores

    """[This function performs binary classification]
    """
    def classification(self, clf_type):
        data = self.transformed_user_responses
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

        # Categorising the labels to binary
        data.iloc[:, 10:207:4] = (data.iloc[:, 10:207:4] == 7.0)

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

            # Split the data into training and test subsets in a stratified fashion
            train_data_question, test_data_question = train_test_split(
                data[relevant_indexes], stratify=data[question_label], test_size=0.3, random_state=42)

            train_x_question = train_data_question.copy()
            test_x_question = test_data_question.copy()

            train_x_question = self.impute_data(train_x_question)
            test_x_question = self.impute_data(test_x_question)

            train_y_question = train_x_question.loc[:, question_label]
            test_y_question = test_x_question.loc[:, question_label]
            train_x_question.drop(columns=question_label, inplace=True)
            test_x_question.drop(columns=question_label, inplace=True)

            train_scaler_question, train_transformer_question, transformed_train_x_question = \
                self.train_data_transformation(train_x_question)

            transformed_test_x_question = \
                self.test_data_transformation(
                    train_scaler_question, train_transformer_question, test_x_question)

            print(clf_type + ' Classification for ' + question + ' Started.')
            clf_score_frame = self.naive_binary_classification(
                transformed_train_x_question, train_y_question, clf_type, question)
            naive_clf_scores = naive_clf_scores.append(clf_score_frame)

            binary_estimators = self.get_binary_estimators()
            validated_estimator = binary_estimators[question]

            best_estimator = validated_estimator[0]
            best_estimator_params = validated_estimator[1]

            best_clf_cross_val_scores, best_estimator_scores = self.run_best_estimator(transformed_train_x_question, train_y_question,
                                                                                       transformed_test_x_question, test_y_question,
                                                                                       best_estimator, best_estimator_params, clf_type, question)
            print(clf_type + ' Classification for ' + question + ' Completed.')
            cross_val_frame = self.create_score_frame_best_estimator(
                best_estimator, best_clf_cross_val_scores)

            cross_val_scores = cross_val_scores.append(cross_val_frame)

            best_clf_frame = self.create_score_frame_best_estimator(
                best_estimator, best_estimator_scores)

            best_clf_scores = best_clf_scores.append(best_clf_frame)

            del relevant_indexes[8:]

        naive_clf_scores = naive_clf_scores.set_index(['Question', 'Model'])
        cross_val_scores = cross_val_scores.set_index(['Question'])
        best_clf_scores = best_clf_scores.set_index(['Question'])

        if not os.path.isdir(BINARY_RESULTS_DIR):
            os.makedirs(BINARY_RESULTS_DIR)

        naive_binary_clf_file_path = os.path.join(
            BINARY_RESULTS_DIR, clf_type + '_naive_undersampled_clf_cross_val_scores_1.1.xlsx')
        best_binary_cross_val_file_path = os.path.join(
            BINARY_RESULTS_DIR, clf_type + '_best_undersampled_clf_cross_val_scores_1.1.xlsx')
        best_binary_clf_file_path = os.path.join(
            BINARY_RESULTS_DIR, clf_type + '_best_undersampled_clf_accuracy_scores_1.1.xlsx')

        # Write results to the EXCEL files
        naive_clf_scores.to_excel(naive_binary_clf_file_path)
        cross_val_scores.to_excel(best_binary_cross_val_file_path)
        best_clf_scores.to_excel(best_binary_clf_file_path)


predictor = TruthfulnessUndersampledPredictor()
