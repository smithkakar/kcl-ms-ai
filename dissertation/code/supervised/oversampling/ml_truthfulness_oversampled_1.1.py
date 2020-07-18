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

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

import warnings

warnings.filterwarnings('ignore')
with warnings.catch_warnings():
    warnings.simplefilter('ignore')

BINARY_RESULTS_DIR = 'results'

# Predictor class for Imbalanced Learning using Oversampling with all features


class TruthfulnessOversamplingPredictor:

    def __init__(self):
        print('Starting Truthfulness Predictor OVERSAMPLED Version 1.1')
        self.user_responses = self.load_survey_data()

        self.user_responses.drop(columns=['Prolific ID'], inplace=True)

        self.reordered_user_responses = self.customise_data()

        self.transformed_user_responses = self.discrete_transform()

        self.binary_classification()

    def __final__(self):
        print('Ending Truthfulness Predictor OVERSAMPLED Version 1.1')

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
        decision_tree_clf = DecisionTreeClassifier(random_state=42)
        logistic_regression_clf = LogisticRegression(
            solver='liblinear', random_state=42)
        svm_clf = svm.SVC(gamma='auto', probability=True, random_state=42)
        knn_clf = KNeighborsClassifier()
        naive_bayes_clf = GaussianNB()
        random_forest_clf = RandomForestClassifier(
            n_estimators=10, random_state=42)
        extra_trees_clf = ExtraTreesClassifier(
            n_estimators=10, random_state=42)
        bagging_clf = BaggingClassifier(random_state=42)
        ada_boost_clf = AdaBoostClassifier(random_state=42)
        xg_boost_clf = XGBClassifier(random_state=42)

        binary_estimators = [decision_tree_clf, logistic_regression_clf, svm_clf, knn_clf, naive_bayes_clf,
                             random_forest_clf, extra_trees_clf, bagging_clf, ada_boost_clf, xg_boost_clf]

        return binary_estimators

    """[This function calculates the cross-validation scores for a question]
    
    Returns:
        [dictionary] -- [cross-val scores]
    """

    def calc_cross_val_scores(self, clf, data, labels, question):
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

        return scores

    """[This function trains the naive binary classifiers]
    
    Returns:
        [list] -- [cross validation scores of all classifiers for a question]
    """

    def run_naive_binary_classifiers(self, train_x, train_y, question):
        naive_binary_estimators = self.get_naive_binary_estimators()
        clf_scores = []

        for estimator in range(len(naive_binary_estimators)):
            clf = naive_binary_estimators[estimator]
            clf.fit(train_x, train_y)
            scores = self.calc_cross_val_scores(
                clf, train_x, train_y, question)
            clf_scores.append(scores)

        return clf_scores

    """[This function creates a dataframe to save scores for naive binary classifiers]
    
    Returns:
        [dataframe] -- [score dataframe for naive binary classifiers]
    """

    def create_score_frame(self, scores):
        dtc_scores, lrc_scores, svm_scores, knn_scores, nbc_scores, rfc_scores, etc_scores, bag_scores, ada_scores, xgb_scores = [
            scores[i] for i in range(len(scores))]

        clf_score_frame = pd.DataFrame({
            'Question': [dtc_scores['Question'], lrc_scores['Question'], svm_scores['Question'], knn_scores['Question'],
                         nbc_scores['Question'],
                         rfc_scores['Question'], etc_scores['Question'], bag_scores['Question'], ada_scores['Question'],
                         xgb_scores['Question']],
            'Model': ['Decision Tree', 'Logistic Regression', 'K-Nearest Neighbors', 'Random Forest', 'Extra Trees',
                      'SVM', 'Naive Bayes', 'Bagging', 'ADABoost', 'XGBoost'],
            'Accuracy': [dtc_scores['Accuracy'], lrc_scores['Accuracy'], knn_scores['Accuracy'],
                         rfc_scores['Accuracy'], etc_scores['Accuracy'], svm_scores['Accuracy'],
                         nbc_scores['Accuracy'], bag_scores['Accuracy'], ada_scores['Accuracy'],
                         xgb_scores['Accuracy']],
            'Balanced Accuracy': [dtc_scores['Balanced Accuracy'], lrc_scores['Balanced Accuracy'],
                                  knn_scores['Balanced Accuracy'],
                                  rfc_scores['Balanced Accuracy'], etc_scores['Balanced Accuracy'],
                                  svm_scores['Balanced Accuracy'],
                                  nbc_scores['Balanced Accuracy'], bag_scores['Balanced Accuracy'],
                                  ada_scores['Balanced Accuracy'],
                                  xgb_scores['Balanced Accuracy']],
            'Precision': [dtc_scores['Precision'], lrc_scores['Precision'], knn_scores['Precision'],
                          rfc_scores['Precision'], etc_scores['Precision'], svm_scores['Precision'],
                          nbc_scores['Precision'], bag_scores['Precision'], ada_scores['Precision'],
                          xgb_scores['Precision']],
            'Recall': [dtc_scores['Recall'], lrc_scores['Recall'], knn_scores['Recall'],
                       rfc_scores['Recall'], etc_scores['Recall'], svm_scores['Recall'],
                       nbc_scores['Recall'], bag_scores['Recall'], ada_scores['Recall'],
                       xgb_scores['Recall']],
            'Specificity': [dtc_scores['Specificity'], lrc_scores['Specificity'], knn_scores['Specificity'],
                            rfc_scores['Specificity'], etc_scores['Specificity'], svm_scores['Specificity'],
                            nbc_scores['Specificity'], bag_scores['Specificity'], ada_scores['Specificity'],
                            xgb_scores['Specificity']],
            'F1': [dtc_scores['F1'], lrc_scores['F1'], knn_scores['F1'],
                   rfc_scores['F1'], etc_scores['F1'], svm_scores['F1'],
                   nbc_scores['F1'], bag_scores['F1'], ada_scores['F1'], xgb_scores['F1']],
            'ROC AUC': [dtc_scores['ROC AUC'], lrc_scores['ROC AUC'], knn_scores['ROC AUC'],
                        rfc_scores['ROC AUC'], etc_scores['ROC AUC'], svm_scores['ROC AUC'],
                        nbc_scores['ROC AUC'], bag_scores['ROC AUC'], ada_scores['ROC AUC'],
                        xgb_scores['ROC AUC']]
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
            columns=['Question', 'Model', 'Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'Specificity', 'F1',
                     'ROC AUC'])

        return clf_score_frame

    """[This function makes a call to run naive binary classifiers]
    
    Returns:
        [dataframe] -- [score dataframe for naive binary classifiers]
    """

    def naive_binary_classification(self, train_x, train_y, question):
        scores = self.run_naive_binary_classifiers(train_x, train_y, question)
        clf_score_frame = self.create_score_frame(scores)

        return clf_score_frame

    """[This function provides the best classifiers manually retrieved from Estimator Finder component for all 50 questions]
    
    Returns:
        [dictionary] -- [instances of best binary classifiers]
    """

    def get_binary_estimators(self):
        # SMOTE cv=10 balanced_accuracy
        binary_estimators = {'Q01': ['SVM', {'C': 100, 'gamma': 'auto', 'kernel': 'rbf'}], 'Q02': ['RandomForestClassifier', {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}], 'Q03': ['KNeighborsClassifier', {'n_neighbors': 1, 'weights': 'uniform'}], 'Q04': ['SVM', {'C': 100, 'gamma': 'auto', 'kernel': 'rbf'}], 'Q05': ['RandomForestClassifier', {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 300}], 'Q06': ['ExtraTreesClassifier', {'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 3, 'min_samples_split': 4, 'n_estimators': 300}], 'Q07': ['SVM', {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}], 'Q08': ['SVM', {'C': 100, 'gamma': 'scale', 'kernel': 'rbf'}], 'Q09': ['KNeighborsClassifier', {'n_neighbors': 1, 'weights': 'uniform'}], 'Q10': ['SVM', {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}], 'Q11': ['SVM', {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}], 'Q12': ['SVM', {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}], 'Q13': ['KNeighborsClassifier', {'n_neighbors': 1, 'weights': 'uniform'}], 'Q14': ['SVM', {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}], 'Q15': ['KNeighborsClassifier', {'n_neighbors': 2, 'weights': 'uniform'}], 'Q16': ['XGBClassifier', {'learning_rate': 0.3, 'max_depth': 2, 'n_estimators': 300}], 'Q17': ['KNeighborsClassifier', {'n_neighbors': 3, 'weights': 'distance'}], 'Q18': ['RandomForestClassifier', {'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 100}], 'Q19': ['RandomForestClassifier', {'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}], 'Q20': ['ExtraTreesClassifier', {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 3, 'min_samples_split': 4, 'n_estimators': 300}], 'Q21': ['XGBClassifier', {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 100}], 'Q22': ['KNeighborsClassifier', {'n_neighbors': 2, 'weights': 'distance'}], 'Q23': ['ExtraTreesClassifier', {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 300}], 'Q24': ['SVM', {
            'C': 100, 'gamma': 'scale', 'kernel': 'rbf'}], 'Q25': ['KNeighborsClassifier', {'n_neighbors': 3, 'weights': 'distance'}], 'Q26': ['SVM', {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}], 'Q27': ['KNeighborsClassifier', {'n_neighbors': 1, 'weights': 'uniform'}], 'Q28': ['RandomForestClassifier', {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}], 'Q29': ['ExtraTreesClassifier', {'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}], 'Q30': ['SVM', {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}], 'Q31': ['KNeighborsClassifier', {'n_neighbors': 1, 'weights': 'uniform'}], 'Q32': ['KNeighborsClassifier', {'n_neighbors': 1, 'weights': 'uniform'}], 'Q33': ['SVM', {'C': 100, 'gamma': 'scale', 'kernel': 'rbf'}], 'Q34': ['KNeighborsClassifier', {'n_neighbors': 1, 'weights': 'uniform'}], 'Q35': ['KNeighborsClassifier', {'n_neighbors': 2, 'weights': 'distance'}], 'Q36': ['KNeighborsClassifier', {'n_neighbors': 1, 'weights': 'uniform'}], 'Q37': ['XGBClassifier', {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 300}], 'Q38': ['XGBClassifier', {'learning_rate': 1, 'max_depth': 2, 'n_estimators': 300}], 'Q39': ['KNeighborsClassifier', {'n_neighbors': 2, 'weights': 'uniform'}], 'Q40': ['KNeighborsClassifier', {'n_neighbors': 1, 'weights': 'uniform'}], 'Q41': ['ExtraTreesClassifier', {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}], 'Q42': ['SVM', {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}], 'Q43': ['KNeighborsClassifier', {'n_neighbors': 1, 'weights': 'uniform'}], 'Q44': ['SVM', {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}], 'Q45': ['KNeighborsClassifier', {'n_neighbors': 1, 'weights': 'uniform'}], 'Q46': ['SVM', {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}], 'Q47': ['ExtraTreesClassifier', {'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 300}], 'Q48': ['KNeighborsClassifier', {'n_neighbors': 2, 'weights': 'distance'}], 'Q49': ['KNeighborsClassifier', {'n_neighbors': 1, 'weights': 'distance'}], 'Q50': ['SVM', {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}]}

        return binary_estimators

    """[This function trains and evaluates the best classifiers for all questions]
    
    Returns:
        [dictionaries] -- [cross-validation and prediction scores]
    """

    def run_best_estimator(self, train_x, train_y, test_x, test_y, estimator, params, question):
        estimator_scores = {}

        if estimator == 'DecisionTreeClassifier':
            clf = DecisionTreeClassifier(max_depth=params['max_depth'], min_samples_split=params['min_samples_split'],
                                         min_samples_leaf=params['min_samples_leaf'], random_state=42)
        elif estimator == 'LogisticRegression':
            clf = LogisticRegression(
                penalty=params['penalty'], C=params['C'], solver=params['solver'], random_state=42)
        elif estimator == 'SVM':
            clf = svm.SVC(C=params['C'], kernel=params['kernel'],
                          gamma=params['gamma'], probability=True, random_state=42)
        elif estimator == 'KNeighborsClassifier':
            clf = KNeighborsClassifier(
                n_neighbors=params['n_neighbors'], weights=params['weights'])
        elif estimator == 'GaussianNB':
            clf = GaussianNB(var_smoothing=params['var_smoothing'])
        elif estimator == 'RandomForestClassifier':
            clf = RandomForestClassifier(criterion=params['criterion'], n_estimators=params['n_estimators'],
                                         max_depth=params['max_depth'], min_samples_split=params['min_samples_split'],
                                         min_samples_leaf=params['min_samples_leaf'], random_state=42)
        elif estimator == 'ExtraTreesClassifier':
            clf = ExtraTreesClassifier(criterion=params['criterion'], n_estimators=params['n_estimators'],
                                       max_depth=params['max_depth'], min_samples_split=params['min_samples_split'],
                                       min_samples_leaf=params['min_samples_leaf'], random_state=42)
        elif estimator == 'BaggingClassifier':
            clf = BaggingClassifier(bootstrap=params['bootstrap'], n_estimators=params['n_estimators'],
                                    max_samples=params['max_samples'], random_state=42)
        elif estimator == 'AdaBoostClassifier':
            clf = AdaBoostClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'],
                                     random_state=42)
        elif estimator == 'XGBClassifier':
            clf = XGBClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'],
                                max_depth=params['max_depth'], random_state=42)

        clf.fit(train_x, train_y)

        cross_val_scores = self.calc_cross_val_scores(
            clf, train_x, train_y, question)

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
        estimator_scores['F1'] = round(f1_score(test_y, predicted_labels), 2)
        estimator_scores['ROC AUC'] = round(
            roc_auc_score(test_y, predicted_prob_true), 2)

        return cross_val_scores, estimator_scores

    """[This function performs binary classification]
    """

    def binary_classification(self):
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
            # Oversampling using SMOTE technique
            sm = SMOTE(random_state=42)
            sampled_train_x_question, sampled_train_y_question = sm.fit_sample(
                train_x_question, train_y_question.ravel())
            sampled_train_x_question = pd.DataFrame(
                sampled_train_x_question, columns=train_x_question.columns)

            train_scaler_question, train_transformer_question, transformed_train_x_question = \
                self.train_data_transformation(sampled_train_x_question)

            transformed_test_x_question = \
                self.test_data_transformation(
                    train_scaler_question, train_transformer_question, test_x_question)

            print('Binary Classification for ' + question + ' Started.')
            clf_score_frame = self.naive_binary_classification(
                transformed_train_x_question, sampled_train_y_question, question)
            naive_clf_scores = naive_clf_scores.append(clf_score_frame)

            binary_estimators = self.get_binary_estimators()
            validated_estimator = binary_estimators[question]

            best_estimator = validated_estimator[0]
            best_estimator_params = validated_estimator[1]

            best_clf_cross_val_scores, best_estimator_scores = self.run_best_estimator(transformed_train_x_question, sampled_train_y_question,
                                                                                       transformed_test_x_question, test_y_question,
                                                                                       best_estimator, best_estimator_params, question)

            print('Binary Classification for ' + question + ' Completed.')

            cross_val_frame = self.create_score_frame_best_estimator(
                best_estimator, best_clf_cross_val_scores)
            cross_val_scores = cross_val_scores.append(cross_val_frame)

            best_clf_frame = self.create_score_frame_best_estimator(
                best_estimator, best_estimator_scores)
            best_clf_scores = best_clf_scores.append(best_clf_frame)

            del relevant_indexes[8:]
            print('Binary Classification for ' + question + ' Completed.')

        naive_clf_scores = naive_clf_scores.set_index(['Question', 'Model'])
        cross_val_scores = cross_val_scores.set_index(['Question'])
        best_clf_scores = best_clf_scores.set_index(['Question'])

        if not os.path.isdir(BINARY_RESULTS_DIR):
            os.makedirs(BINARY_RESULTS_DIR)

        naive_binary_clf_file_path = os.path.join(
            BINARY_RESULTS_DIR, 'naive_oversampled_clf_cross_val_scores_1.1.xlsx')
        best_binary_cross_val_file_path = os.path.join(
            BINARY_RESULTS_DIR, 'best_oversampled_clf_cross_val_scores_1.1.xlsx')
        best_binary_clf_file_path = os.path.join(
            BINARY_RESULTS_DIR, 'best_oversampled_clf_accuracy_scores_1.1.xlsx')
        # Write results to the EXCEL files
        naive_clf_scores.to_excel(naive_binary_clf_file_path)
        cross_val_scores.to_excel(best_binary_cross_val_file_path)
        best_clf_scores.to_excel(best_binary_clf_file_path)


predictor = TruthfulnessOversamplingPredictor()
