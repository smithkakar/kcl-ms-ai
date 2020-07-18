#!/usr/bin/env python3
# coding: utf-8

import os
import sys
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
from sklearn.ensemble import ExtraTreesClassifier

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

from imblearn.metrics import geometric_mean_score

import warnings

warnings.filterwarnings('ignore')
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

RESULTS_DIR = 'results'

# Predictor class for Cost-Sensitive Learning with informative features


class TruthfulnessCostSensitivePredictor:

    def __init__(self):
        try:
            print("Starting Truthfulness Predictor COST-SENSITIVE Version 1.4")
            self.user_responses = self.load_survey_data()

            self.user_responses.drop(columns=['Prolific ID'], inplace=True)

            self.reordered_user_responses = self.customise_data()

            self.transformed_user_responses = self.discrete_transform()
            print('Binary Classification Started.')
            self.classification('binary')
            print('Binary Classification Completed.')
            print('Multi-class Classification Started.')
            self.classification('multi-class')
            print('Multi-class Classification Completed.')
        except Exception as e:
            print('Error: ', str(e))
            sys.exit(1)

    def __final__(self):
        print("Ending Truthfulness Predictor COST-SENSITIVE Version 1.4")

    """[This function loads the data from csv file to pandas dataframe]

    Returns:
        [dataframe] -- [data]
    """

    def load_survey_data(self):
        try:
            df = pd.read_csv('All_Responses_Removed.csv')
        except IOError as e:
            print('Problem occured while loading the csv file. Please place the file at the same location along with this script.')
            raise
        except Exception as e:
            print('Problem occured while loading the csv file.')
            print('Error: ', str(e))
            raise

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

    def get_naive_binary_estimators(self, class_weights):
        decision_tree_clf = DecisionTreeClassifier(
            class_weight=class_weights, random_state=42)
        logistic_regression_clf = LogisticRegression(
            solver='liblinear', class_weight=class_weights, random_state=42)
        svm_clf = svm.SVC(gamma='auto', probability=True,
                          class_weight=class_weights, random_state=42)
        random_forest_clf = RandomForestClassifier(
            n_estimators=10, class_weight=class_weights, random_state=42)
        extra_trees_clf = ExtraTreesClassifier(
            n_estimators=10, class_weight=class_weights, random_state=42)
        balanced_rf_clf = BalancedRandomForestClassifier(
            sampling_strategy='not majority', class_weight=class_weights, random_state=42)

        naive_estimators = [decision_tree_clf, logistic_regression_clf,
                            svm_clf, random_forest_clf, extra_trees_clf, balanced_rf_clf]

        return naive_estimators

    """[This function initialises the naive multiclass models]

    Returns:
        [list] -- [instances of naive multiclass models]
    """

    def get_naive_multi_class_estimators(self, class_weights):
        decision_tree_clf = DecisionTreeClassifier(
            class_weight=class_weights, random_state=42)
        # Measure using cross-entropy loss
        multinomial_lr_clf = LogisticRegression(
            solver='lbfgs', multi_class='multinomial', class_weight=class_weights, random_state=42)
        svm_clf = svm.SVC(gamma='auto', class_weight=class_weights,
                          probability=True, random_state=42)
        random_forest_clf = RandomForestClassifier(
            n_estimators=10, class_weight=class_weights, random_state=42)
        extra_trees_clf = ExtraTreesClassifier(
            n_estimators=10, class_weight=class_weights, random_state=42)
        lr_ovr_clf = LogisticRegression(
            solver='lbfgs', multi_class='ovr',  class_weight=class_weights, random_state=42)
        balanced_rf_clf = BalancedRandomForestClassifier(
            sampling_strategy='not majority', class_weight=class_weights, random_state=42)
        multi_class_estimators = [decision_tree_clf, multinomial_lr_clf, svm_clf,
                                  random_forest_clf, extra_trees_clf, lr_ovr_clf, balanced_rf_clf]

        return multi_class_estimators

    """[This function calculates the cross-validation scores for a question]

    Returns:
        [dictionary] -- [cross-val scores]
    """

    def calc_cross_val_scores(self, clf, data, labels, clf_type, question):
        scores = {}

        try:
            predicted_scores = cross_val_score(
                clf, data, labels, cv=10, scoring="accuracy")
        except Exception as e:
            raise ValueError(e)

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
        if clf_type == 'binary':
            naive_estimators = self.get_naive_binary_estimators(class_weights)
        if clf_type == 'multi-class':
            naive_estimators = self.get_naive_multi_class_estimators(
                class_weights)

        clf_scores = []
        for estimator in range(len(naive_estimators)):
            clf = naive_estimators[estimator]
            clf.fit(train_x, train_y)
            try:
                scores = self.calc_cross_val_scores(
                    clf, train_x, train_y, clf_type, question)
            except ValueError as e:
                raise ValueError(e)
            clf_scores.append(scores)

        # print('Perfect Confusion Matrix for Q-%s is: ' % (str(question).zfill(2)))
        # perfect_labels = train_y
        # print(confusion_matrix(train_y, perfect_labels))

        return clf_scores

    """[This function creates a dataframe to save scores for naive classifiers]

    Returns:
        [dataframe] -- [score dataframe for naive classifiers]
    """

    def create_score_frame(self, scores):
        dtc_scores, lrc_scores, svm_scores, rfc_scores, etc_scores, balanced_rf_scores = [
            scores[i] for i in range(len(scores))]

        clf_score_frame = pd.DataFrame({
            'Question': [dtc_scores['Question'], lrc_scores['Question'], svm_scores['Question'],
                         rfc_scores['Question'], etc_scores['Question'], balanced_rf_scores['Question']],
            'Model': ['Decision Tree', 'Logistic Regression', 'SVM', 'Random Forest', 'Extra Trees',
                      'Balanced Random Forest'],
            'Accuracy': [dtc_scores['Accuracy'], lrc_scores['Accuracy'], svm_scores['Accuracy'],
                         rfc_scores['Accuracy'], etc_scores['Accuracy'], balanced_rf_scores['Accuracy']],
            'Balanced Accuracy': [dtc_scores['Balanced Accuracy'], lrc_scores['Balanced Accuracy'],
                                  svm_scores['Balanced Accuracy'],
                                  rfc_scores['Balanced Accuracy'], etc_scores['Balanced Accuracy'],
                                  balanced_rf_scores['Balanced Accuracy']],
            'Precision': [dtc_scores['Precision'], lrc_scores['Precision'], svm_scores['Precision'],
                          rfc_scores['Precision'], etc_scores['Precision'], balanced_rf_scores['Precision']],
            'Recall': [dtc_scores['Recall'], lrc_scores['Recall'], svm_scores['Recall'],
                       rfc_scores['Recall'], etc_scores['Recall'], balanced_rf_scores['Recall']],
            'Specificity': [dtc_scores['Specificity'], lrc_scores['Specificity'], svm_scores['Specificity'],
                            rfc_scores['Specificity'], etc_scores['Specificity'], balanced_rf_scores['Specificity']],
            'F1': [dtc_scores['F1'], lrc_scores['F1'], svm_scores['F1'],
                   rfc_scores['F1'], etc_scores['F1'], balanced_rf_scores['F1']],
            'ROC AUC': [dtc_scores['ROC AUC'], lrc_scores['ROC AUC'], svm_scores['ROC AUC'],
                        rfc_scores['ROC AUC'], etc_scores['ROC AUC'], balanced_rf_scores['ROC AUC']]
        }, columns=['Question', 'Model', 'Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 'ROC AUC'])

        return clf_score_frame

    """[This function creates a dataframe to save multiclassificaiton scores]

    Returns:
        [dataframe] -- [score dataframe for naive classifiers]
    """

    def create_multi_class_score_frame(self, scores):
        decision_tree_clf, multinomial_lr_clf, svm_clf, random_forest_clf, extra_trees_clf, lr_ovr_clf, balanced_rf_clf = [
            scores[i] for i in range(len(scores))]

        clf_score_frame = pd.DataFrame({
            'Question': [decision_tree_clf['Question'], multinomial_lr_clf['Question'], svm_clf['Question'],
                         random_forest_clf['Question'], extra_trees_clf['Question'],
                         lr_ovr_clf['Question'], balanced_rf_clf['Question']],
            'Model': ['Decision Tree', 'Multinomial Logistic Regression', 'SVM', 'Random Forest', 'Extra Trees',
                      'Logistic Regression OVR', 'Balanced Random Forest'],
            'Accuracy': [decision_tree_clf['Accuracy'], multinomial_lr_clf['Accuracy'], svm_clf['Accuracy'],
                         random_forest_clf['Accuracy'], extra_trees_clf['Accuracy'],
                         lr_ovr_clf['Accuracy'], balanced_rf_clf['Accuracy']],
            'Balanced Accuracy': [decision_tree_clf['Balanced Accuracy'], multinomial_lr_clf['Balanced Accuracy'], svm_clf['Balanced Accuracy'],
                                  random_forest_clf['Balanced Accuracy'], extra_trees_clf['Balanced Accuracy'],
                                  lr_ovr_clf['Balanced Accuracy'], balanced_rf_clf['Balanced Accuracy']],
            'Precision': [decision_tree_clf['Precision'], multinomial_lr_clf['Precision'], svm_clf['Precision'],
                          random_forest_clf['Precision'], extra_trees_clf['Precision'],
                          lr_ovr_clf['Precision'], balanced_rf_clf['Precision']],
            'Recall': [decision_tree_clf['Recall'], multinomial_lr_clf['Recall'], svm_clf['Recall'],
                       random_forest_clf['Recall'], extra_trees_clf['Recall'],
                       lr_ovr_clf['Recall'], balanced_rf_clf['Recall']],
            'Specificity': [decision_tree_clf['Specificity'], multinomial_lr_clf['Specificity'], svm_clf['Specificity'],
                            random_forest_clf['Specificity'], extra_trees_clf['Specificity'],
                            lr_ovr_clf['Specificity'], balanced_rf_clf['Specificity']],
            'F1': [decision_tree_clf['F1'], multinomial_lr_clf['F1'], svm_clf['F1'],
                   random_forest_clf['F1'], extra_trees_clf['F1'],
                   lr_ovr_clf['F1'], balanced_rf_clf['F1']],
            'ROC AUC': [decision_tree_clf['ROC AUC'], multinomial_lr_clf['ROC AUC'], svm_clf['ROC AUC'],
                        random_forest_clf['ROC AUC'], extra_trees_clf['ROC AUC'],
                        lr_ovr_clf['ROC AUC'], balanced_rf_clf['ROC AUC']],
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
        try:
            scores = self.run_naive_classifiers(
                train_x, train_y, class_weights, clf_type, question)
        except ValueError as e:
            raise ValueError(e)

        if clf_type == 'binary':
            clf_score_frame = self.create_score_frame(scores)
        if clf_type == 'multi-class':
            clf_score_frame = self.create_multi_class_score_frame(scores)

        return clf_score_frame

    """[This function provides the best binary classifiers manually retrieved from Estimator Finder component for all 50 questions]
    
    Returns:
        [dictionary] -- [instances of best binary classifiers]
    """

    def get_binary_estimators(self):

        # cv=5 balanced_accuracy
        binary_estimators = {'Q01': ['SVM', {'C': 0.01, 'class_weight': {0: 0.643652561247216, 1: 2.24031007751938}, 'gamma': 'auto', 'kernel': 'rbf'}], 'Q02': ['ExtraTreesClassifier', {'class_weight': {0: 0.8376811594202899, 1: 1.240343347639485}, 'criterion': 'entropy', 'max_depth': 2, 'min_samples_leaf': 3, 'min_samples_split': 4, 'n_estimators': 100}], 'Q03': ['DecisionTreeClassifier', {'class_weight': {0: 0.7225, 1: 1.6235955056179776}, 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2}], 'Q04': ['DecisionTreeClassifier', {'class_weight': {0: 0.553639846743295, 1: 5.160714285714286}, 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2}], 'Q05': ['RandomForestClassifier', {'class_weight': {0: 0.7506493506493507, 1: 1.4974093264248705}, 'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 2, 'min_samples_split': 3, 'n_estimators': 300}], 'Q06': ['ExtraTreesClassifier', {'class_weight': {0: 0.7391304347826086, 1: 1.5454545454545454}, 'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 300}], 'Q07': ['DecisionTreeClassifier', {'class_weight': {0: 0.6509009009009009, 1: 2.156716417910448}, 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2}], 'Q08': ['BalancedRandomForestClassifier', {'class_weight': {0: 0.5838383838383838, 1: 3.4819277108433737}, 'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 100, 'sampling_strategy': 'majority'}], 'Q09': ['BalancedRandomForestClassifier', {'class_weight': {0: 0.6658986175115207, 1: 2.0069444444444446}, 'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q10': ['BalancedRandomForestClassifier', {'class_weight': {0: 0.664367816091954, 1: 2.020979020979021}, 'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'not majority'}], 'Q11': ['BalancedRandomForestClassifier', {'class_weight': {0: 0.608421052631579, 1: 2.8058252427184467}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 300, 'sampling_strategy': 'majority'}], 'Q12': ['BalancedRandomForestClassifier', {'class_weight': {0: 0.5494296577946768, 1: 5.5576923076923075}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 300, 'sampling_strategy': 'majority'}], 'Q13': ['SVM', {'C': 0.01, 'class_weight': {0: 0.5995850622406639, 1: 3.0104166666666665}, 'gamma': 'scale', 'kernel': 'linear'}], 'Q14': ['BalancedRandomForestClassifier', {'class_weight': {0: 0.5946502057613169, 1: 3.141304347826087}, 'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 100, 'sampling_strategy': 'not majority'}], 'Q15': ['ExtraTreesClassifier', {'class_weight': {0: 0.578, 1: 3.7051282051282053}, 'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 100}], 'Q16': ['ExtraTreesClassifier', {'class_weight': {0: 0.8328530259365994, 1: 1.251082251082251}, 'criterion': 'entropy', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 100}], 'Q17': ['BalancedRandomForestClassifier', {'class_weight': {0: 0.7118226600985221, 1: 1.680232558139535}, 'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 100, 'sampling_strategy': 'majority'}], 'Q18': ['RandomForestClassifier', {'class_weight': {0: 0.7853260869565217, 1: 1.3761904761904762}, 'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 300}], 'Q19': ['RandomForestClassifier', {'class_weight': {0: 0.7896174863387978, 1: 1.3632075471698113}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}], 'Q20': ['DecisionTreeClassifier', {'class_weight': {0: 0.7545691906005222, 1: 1.4820512820512821}, 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2}], 'Q21': ['RandomForestClassifier', {'class_weight': {0: 0.71712158808933, 1: 1.6514285714285715}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 3, 'min_samples_split': 3, 'n_estimators': 100}], 'Q22': ['DecisionTreeClassifier', {'class_weight': {0: 0.6816037735849056, 1: 1.8766233766233766}, 'max_depth': 6, 'min_samples_leaf': 3, 'min_samples_split': 4}], 'Q23': ['ExtraTreesClassifier', {'class_weight': {0: 0.7526041666666666, 1: 1.4896907216494846}, 'criterion': 'entropy', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}], 'Q24': ['ExtraTreesClassifier', {'class_weight': {0: 0.71712158808933, 1: 1.6514285714285715}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 300}], 'Q25': ['RandomForestClassifier', {'class_weight': {0: 0.7810810810810811, 1: 1.3894230769230769}, 'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 3, 'min_samples_split': 4,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       'n_estimators': 300}], 'Q26': ['RandomForestClassifier', {'class_weight': {0: 0.6162046908315565, 1: 2.6513761467889907}, 'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}], 'Q27': ['SVM', {'C': 0.01, 'class_weight': {0: 0.7316455696202532, 1: 1.5792349726775956}, 'gamma': 'auto', 'kernel': 'rbf'}], 'Q28': ['DecisionTreeClassifier', {'class_weight': {0: 0.7983425414364641, 1: 1.337962962962963}, 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 3}], 'Q29': ['ExtraTreesClassifier', {'class_weight': {0: 0.7206982543640897, 1: 1.6327683615819208}, 'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 100}], 'Q30': ['BalancedRandomForestClassifier', {'class_weight': {0: 0.6282608695652174, 1: 2.4491525423728815}, 'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q31': ['BalancedRandomForestClassifier', {'class_weight': {0: 0.6598173515981736, 1: 2.0642857142857145}, 'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'majority'}], 'Q32': ['ExtraTreesClassifier', {'class_weight': {0: 0.7189054726368159, 1: 1.6420454545454546}, 'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}], 'Q33': ['ExtraTreesClassifier', {'class_weight': {0: 0.7585301837270341, 1: 1.467005076142132}, 'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 300}], 'Q34': ['DecisionTreeClassifier', {'class_weight': {0: 0.6736596736596736, 1: 1.9395973154362416}, 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2}], 'Q35': ['ExtraTreesClassifier', {'class_weight': {0: 0.6752336448598131, 1: 1.9266666666666667}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 3, 'min_samples_split': 4, 'n_estimators': 100}], 'Q36': ['BalancedRandomForestClassifier', {'class_weight': {0: 0.6930455635491607, 1: 1.795031055900621}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 300, 'sampling_strategy': 'auto'}], 'Q37': ['DecisionTreeClassifier', {'class_weight': {0: 0.7297979797979798, 1: 1.5879120879120878}, 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2}], 'Q38': ['ExtraTreesClassifier', {'class_weight': {0: 0.7768817204301075, 1: 1.4029126213592233}, 'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 300}], 'Q39': ['BalancedRandomForestClassifier', {'class_weight': {0: 0.6241900647948164, 1: 2.5130434782608697}, 'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q40': ['DecisionTreeClassifier', {'class_weight': {0: 0.6705336426914154, 1: 1.965986394557823}, 'max_depth': 4, 'min_samples_leaf': 3, 'min_samples_split': 3}], 'Q41': ['ExtraTreesClassifier', {'class_weight': {0: 0.7225, 1: 1.6235955056179776}, 'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 100}], 'Q42': ['ExtraTreesClassifier', {'class_weight': {0: 0.706601466992665, 1: 1.7100591715976332}, 'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 100}], 'Q43': ['SVM', {'C': 0.5, 'class_weight': {0: 0.6658986175115207, 1: 2.0069444444444446}, 'gamma': 'auto', 'kernel': 'rbf'}], 'Q44': ['ExtraTreesClassifier', {'class_weight': {0: 0.76657824933687, 1: 1.4378109452736318}, 'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 3, 'min_samples_split': 3, 'n_estimators': 100}], 'Q45': ['BalancedRandomForestClassifier', {'class_weight': {0: 0.6816037735849056, 1: 1.8766233766233766}, 'criterion': 'entropy', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q46': ['BalancedRandomForestClassifier', {'class_weight': {0: 0.6162046908315565, 1: 2.6513761467889907}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q47': ['SVM', {'C': 0.01, 'class_weight': {0: 0.7605263157894737, 1: 1.4595959595959596}, 'gamma': 'scale', 'kernel': 'rbf'}], 'Q48': ['DecisionTreeClassifier', {'class_weight': {0: 0.7686170212765957, 1: 1.4306930693069306}, 'max_depth': 4, 'min_samples_leaf': 3, 'min_samples_split': 4}], 'Q49': ['BalancedRandomForestClassifier', {'class_weight': {0: 0.6930455635491607, 1: 1.795031055900621}, 'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'auto'}], 'Q50': ['BalancedRandomForestClassifier', {'class_weight': {0: 0.5381750465549349, 1: 7.048780487804878}, 'criterion': 'entropy', 'max_depth': 2, 'min_samples_leaf': 2, 'min_samples_split': 3, 'n_estimators': 300, 'sampling_strategy': 'not majority'}]}

        return binary_estimators

    """[This function provides the best multiclass models manually retrieved from Estimator Finder component for all 50 questions]
    
    Returns:
        [dictionary] -- [instances of best binary classifiers]
    """

    def get_multi_class_estimators(self):

        # cv=5 f1_micro
        multi_class_estimators = {'Q01': ['BalancedRandomForestClassifier', {'class_weight': {0: 1.4183006535947713, 1: 1.954954954954955, 2: 0.5607235142118863}, 'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 3, 'min_samples_split': 4, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q02': ['BalancedRandomForestClassifier', {'class_weight': {0: 27.0, 1: 11.571428571428571, 2: 0.34763948497854075}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 3, 'min_samples_split': 4, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q03': ['BalancedRandomForestClassifier', {'class_weight': {0: 2.212962962962963, 1: 3.0641025641025643, 2: 0.4500941619585687}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q04': ['BalancedRandomForestClassifier', {'class_weight': {0: 0.4927536231884058, 1: 3.7777777777777777, 2: 1.4166666666666667}, 'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q05': ['BalancedRandomForestClassifier', {'class_weight': {0: 14.066666666666666, 1: 5.410256410256411, 2: 0.3644214162348877}, 'criterion': 'entropy', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 100, 'sampling_strategy': 'not majority'}], 'Q06': ['BalancedRandomForestClassifier', {'class_weight': {0: 5.5476190476190474, 1: 2.5053763440860215, 2: 0.41312056737588654}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 3, 'min_samples_split': 4, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q07': ['BalancedRandomForestClassifier', {'class_weight': {0: 1.5, 1: 2.1176470588235294, 2: 0.5373134328358209}, 'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 2, 'min_samples_split': 3, 'n_estimators': 100, 'sampling_strategy': 'not majority'}], 'Q08': ['ExtraTreesClassifier', {'class_weight': {0: 0.7376543209876543, 1: 1.6597222222222223, 2: 0.9598393574297188}, 'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 300}], 'Q09': ['RandomForestClassifier', {'class_weight': {0: 2.0277777777777777, 1: 1.8717948717948718, 2: 0.5069444444444444}, 'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 2, 'min_samples_split': 3, 'n_estimators': 100}], 'Q10': ['RandomForestClassifier', {'class_weight': {0: 1.4303030303030304, 1: 2.017094017094017, 2: 0.5539906103286385}, 'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 100}], 'Q11': ['SVM', {'C': 0.1, 'class_weight': {0: 0.9907407407407407, 1: 1.829059829059829, 2: 0.6925566343042071}, 'gamma': 'scale', 'kernel': 'linear'}], 'Q12': ['BalancedRandomForestClassifier', {'class_weight': {0: 0.5042735042735043, 1: 2.8095238095238093, 2: 1.5128205128205128}, 'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q13': ['SVM', {'C': 1, 'class_weight': {0: 0.8090277777777778, 1: 1.8943089430894309, 2: 0.8090277777777778}, 'gamma': 'scale', 'kernel': 'rbf'}], 'Q14': ['SVM', {'C': 10, 'class_weight': {0: 0.7181818181818181, 1: 2.1944444444444446, 2: 0.8681318681318682}, 'gamma': 'auto', 'kernel': 'linear'}], 'Q15': ['RandomForestClassifier', {'class_weight': {0: 0.6125356125356125, 1: 3.4126984126984126, 2: 0.9307359307359307}, 'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 300}], 'Q16': ['RandomForestClassifier', {'class_weight': {0: 27.444444444444443, 1: 6.333333333333333, 2: 0.3564213564213564}, 'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 100}], 'Q17': ['RandomForestClassifier', {'class_weight': {0: 19.416666666666668, 1: 1.3625730994152048, 2: 0.45155038759689925}, 'criterion': 'entropy', 'max_depth': 2, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}], 'Q18': ['BalancedRandomForestClassifier', {'class_weight': {0: 19.916666666666668, 1: 3.1866666666666665, 2: 0.37936507936507935}, 'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 100, 'sampling_strategy': 'not majority'}], 'Q19': ['BalancedRandomForestClassifier', {'class_weight': {0: 9.958333333333334, 1: 4.192982456140351, 2: 0.3757861635220126}, 'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 100, 'sampling_strategy': 'not majority'}], 'Q20': ['BalancedRandomForestClassifier', {'class_weight': {0: 18.083333333333332, 1: 4.018518518518518, 2: 0.37094017094017095}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 3, 'min_samples_split': 4, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q21': ['BalancedRandomForestClassifier', {'class_weight': {0: 9.25, 1: 1.8974358974358974, 2: 0.4228571428571429}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 3, 'min_samples_split': 4, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q22': ['RandomForestClassifier', {'class_weight': {0: 5.041666666666667, 1: 1.1203703703703705, 2: 0.5238095238095238}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}], 'Q23': ['BalancedRandomForestClassifier', {'class_weight': {0: 24.22222222222222, 1: 3.303030303030303, 2: 0.3765112262521589}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q24': ['BalancedRandomForestClassifier', {'class_weight': {0: 4.2105263157894735, 1: 1.7391304347826086, 2: 0.45714285714285713}, 'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q25': ['BalancedRandomForestClassifier', {'class_weight': {0: 19.833333333333332, 1: 2.9382716049382718, 2: 0.3832528180354267}, 'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 2,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              'min_samples_split': 4, 'n_estimators': 100, 'sampling_strategy': 'not majority'}], 'Q26': ['ExtraTreesClassifier', {'class_weight': {0: 2.3225806451612905, 1: 0.9473684210526315, 2: 0.6605504587155964}, 'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 100}], 'Q27': ['BalancedRandomForestClassifier', {'class_weight': {0: 11.380952380952381, 1: 1.6258503401360545, 2: 0.4353369763205829}, 'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q29': ['BalancedRandomForestClassifier', {'class_weight': {0: 15.166666666666666, 1: 2.757575757575758, 2: 0.3888888888888889}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q30': ['RandomForestClassifier', {'class_weight': {0: 1.0085470085470085, 1: 1.9666666666666666, 2: 0.6666666666666666}, 'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 100}], 'Q31': ['ExtraTreesClassifier', {'class_weight': {0: 4.016666666666667, 1: 0.9917695473251029, 2: 0.5738095238095238}, 'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 100}], 'Q32': ['BalancedRandomForestClassifier', {'class_weight': {0: 7.212121212121212, 1: 1.5555555555555556, 2: 0.45075757575757575}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 100, 'sampling_strategy': 'not majority'}], 'Q33': ['BalancedRandomForestClassifier', {'class_weight': {0: 24.666666666666668, 1: 3.3636363636363638, 2: 0.3756345177664975}, 'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'not majority'}], 'Q34': ['RandomForestClassifier', {'class_weight': {0: 7.833333333333333, 1: 1.030701754385965, 2: 0.5257270693512305}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 100}], 'Q35': ['ExtraTreesClassifier', {'class_weight': {0: 7.151515151515151, 1: 1.048888888888889, 2: 0.5244444444444445}, 'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 100}], 'Q36': ['BalancedRandomForestClassifier', {'class_weight': {0: 7.066666666666666, 1: 1.7235772357723578, 2: 0.4389233954451346}, 'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q37': ['RandomForestClassifier', {'class_weight': {0: 9.916666666666666, 1: 1.6527777777777777, 2: 0.4358974358974359}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}], 'Q38': ['BalancedRandomForestClassifier', {'class_weight': {0: 19.583333333333332, 1: 3.1333333333333333, 2: 0.3802588996763754}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'not majority'}], 'Q39': ['RandomForestClassifier', {'class_weight': {0: 3.0641025641025643, 1: 0.8129251700680272, 2: 0.6927536231884058}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 100}], 'Q40': ['BalancedRandomForestClassifier', {'class_weight': {0: 7.933333333333334, 1: 0.9794238683127572, 2: 0.5396825396825397}, 'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q41': ['RandomForestClassifier', {'class_weight': {0: 19.833333333333332, 1: 1.391812865497076, 2: 0.448210922787194}, 'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}], 'Q42': ['RandomForestClassifier', {'class_weight': {0: 36.166666666666664, 1: 1.5724637681159421, 2: 0.4280078895463511}, 'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 100}], 'Q43': ['BalancedRandomForestClassifier', {'class_weight': {0: 6.636363636363637, 1: 1.140625, 2: 0.5069444444444444}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 100, 'sampling_strategy': 'not majority'}], 'Q44': ['BalancedRandomForestClassifier', {'class_weight': {0: 12.944444444444445, 1: 2.9871794871794872, 2: 0.3864013266998342}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'not majority'}], 'Q45': ['RandomForestClassifier', {'class_weight': {0: 26.444444444444443, 1: 0.967479674796748, 2: 0.5185185185185185}, 'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 3, 'min_samples_split': 4, 'n_estimators': 100}], 'Q46': ['RandomForestClassifier', {'class_weight': {0: 1.0909090909090908, 1: 1.7560975609756098, 2: 0.6605504587155964}, 'criterion': 'entropy', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}], 'Q47': ['RandomForestClassifier', {'class_weight': {0: 10.0, 1: 2.2857142857142856, 2: 0.40609137055837563}, 'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 100}], 'Q48': ['BalancedRandomForestClassifier', {'class_weight': {0: 26.22222222222222, 1: 2.5376344086021505, 2: 0.38943894389438943}, 'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300, 'sampling_strategy': 'not majority'}], 'Q49': ['RandomForestClassifier', {'class_weight': {0: 6.611111111111111, 1: 1.2395833333333333, 2: 0.4897119341563786}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 300}], 'Q50': ['BalancedRandomForestClassifier', {'class_weight': {0: 0.4527938342967245, 1: 3.7301587301587302, 2: 1.910569105691057}, 'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'sampling_strategy': 'not majority'}]}

        return multi_class_estimators

    """[This function trains and evaluates the best classifiers for all questions]
    
    Returns:
        [dictionaries] -- [cross-validation and prediction scores]
    """

    def run_best_estimator(self, train_x, train_y, test_x, test_y, estimator, params, clf_type, question):
        estimator_scores = {}

        if estimator == 'DecisionTreeClassifier':
            clf = DecisionTreeClassifier(max_depth=params['max_depth'], min_samples_split=params['min_samples_split'],
                                         min_samples_leaf=params['min_samples_leaf'], class_weight=params['class_weight'], random_state=42)
        elif estimator == 'LogisticRegression':
            clf = LogisticRegression(
                C=params['C'], solver=params['solver'], class_weight=params['class_weight'], random_state=42)
        elif estimator == 'SVM':
            clf = svm.SVC(C=params['C'], kernel=params['kernel'],
                          gamma=params['gamma'], probability=True, class_weight=params['class_weight'], random_state=42)
        elif estimator == 'RandomForestClassifier':
            clf = RandomForestClassifier(criterion=params['criterion'], n_estimators=params['n_estimators'],
                                         max_depth=params['max_depth'], min_samples_split=params['min_samples_split'],
                                         min_samples_leaf=params['min_samples_leaf'], class_weight=params['class_weight'], random_state=42)
        elif estimator == 'ExtraTreesClassifier':
            clf = ExtraTreesClassifier(criterion=params['criterion'], n_estimators=params['n_estimators'],
                                       max_depth=params['max_depth'], min_samples_split=params['min_samples_split'],
                                       min_samples_leaf=params['min_samples_leaf'], class_weight=params['class_weight'], random_state=42)
        elif estimator == 'BalancedRandomForestClassifier':
            clf = BalancedRandomForestClassifier(n_estimators=params['n_estimators'],
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
        computed_weights = []
        try:
            demographics_column_indexes = ['Age', 'Gender', 'IUIPC-Awareness', 'IUIPC-Collection', 'IUIPC-Control',
                                           'Online-Presence', 'Personal-Stability', 'Reciprocity']
            relevant_indexes.extend(demographics_column_indexes)

            naive_clf_scores = pd.DataFrame(columns=[
                                            'Question', 'Model', 'Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 'ROC AUC'])
            cross_val_scores = pd.DataFrame(columns=[
                                            'Question', 'Model', 'Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 'ROC AUC'])
            best_clf_scores = pd.DataFrame(
                columns=['Question', 'Model', 'Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 'ROC AUC'])

            data = self.transformed_user_responses.copy()
            if clf_type == 'binary':
                data.iloc[:, 10:207:4] = (data.iloc[:, 10:207:4] == 7.0)
            if clf_type == 'multi-class':
                data.iloc[:, 10:207:4] = data.iloc[:,
                                                   10:207:4].replace([1.0, 2.0, 3.0], 0)
                data.iloc[:, 10:207:4] = data.iloc[:,
                                                   10:207:4].replace([4.0, 5.0, 6.0], 1)
                data.iloc[:, 10:207:4] = data.iloc[:,
                                                   10:207:4].replace([7.0], 2)

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

                question_label = str(question_number).zfill(
                    2) + '-Truthfulness'

                try:
                    if clf_type == 'binary':
                        train_data_question, test_data_question = train_test_split(
                            data[relevant_indexes], stratify=data[question_label], test_size=0.3,
                            random_state=42)

                    if clf_type == 'multi-class':
                        cleaned_user_responses = data.loc[:, relevant_indexes].dropna(
                        )
                        train_data_question, test_data_question = train_test_split(cleaned_user_responses,
                                                                                   stratify=cleaned_user_responses[
                                                                                       question_label],
                                                                                   test_size=0.3, random_state=42)
                except Exception as e:
                    print('Error: ' + str(e))
                    print('Record skipped.')
                    del relevant_indexes[8:]
                    continue

                train_x_question = train_data_question.copy()
                test_x_question = test_data_question.copy()

                if clf_type == 'binary':
                    train_x_question = self.impute_data(train_x_question)
                    test_x_question = self.impute_data(test_x_question)

                    computed_weights = class_weight.compute_class_weight('balanced', np.unique(
                        train_x_question[question_label]), train_x_question[question_label]).tolist()
                    class_weights[question] = {
                        0: computed_weights[0], 1: computed_weights[1]}

                try:
                    if clf_type == 'multi-class':
                        computed_weights = class_weight.compute_class_weight('balanced', np.unique(
                            train_x_question[question_label]), train_x_question[question_label]).tolist()
                        class_weights[question] = {
                            0: computed_weights[0], 1: computed_weights[1], 2: computed_weights[2]}
                except Exception as e:
                    print("Error: " + str(e))
                    print('Record skipped.')
                    del relevant_indexes[8:]
                    continue

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
                try:
                    clf_score_frame = self.naive_classification(featured_train_data, train_y_question,
                                                                class_weights[question], clf_type, question)
                except ValueError as e:
                    print("Value Error: " + str(e))
                    print('Record skipped.')
                    del relevant_indexes[8:]
                    continue
                naive_clf_scores = naive_clf_scores.append(clf_score_frame)

                if clf_type == 'binary':
                    binary_estimators = self.get_binary_estimators()
                    validated_estimator = binary_estimators[question]
                if clf_type == 'multi-class':
                    multi_class_estimators = self.get_multi_class_estimators()
                    validated_estimator = multi_class_estimators[question]

                best_estimator = validated_estimator[0]
                best_estimator_params = validated_estimator[1]
                try:
                    best_clf_cross_val_scores, best_estimator_scores = self.run_best_estimator(featured_train_data, train_y_question,
                                                                                               featured_test_data, test_y_question,
                                                                                               best_estimator, best_estimator_params, clf_type, question)
                except ValueError as e:
                    print("Value Error: " + str(e))
                    print('Record skipped.')
                    del relevant_indexes[8:]
                    continue
                print(clf_type + ' Classification for ' +
                      question + ' Completed.')

                cross_val_frame = self.create_score_frame_best_estimator(
                    best_estimator, best_clf_cross_val_scores)

                cross_val_scores = cross_val_scores.append(cross_val_frame)

                best_clf_frame = self.create_score_frame_best_estimator(
                    best_estimator, best_estimator_scores)

                best_clf_scores = best_clf_scores.append(best_clf_frame)

                del relevant_indexes[8:]

        except Exception as e:
            print('Problem occured during classification.')
            raise

        naive_clf_scores = naive_clf_scores.set_index(['Question', 'Model'])
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
            RESULTS_DIR, clf_type + '_naive_cost_sensitive_clf_cross_val_scores_1.4.xlsx')
        best_cross_val_file_path = os.path.join(
            RESULTS_DIR, clf_type + '_best_clf_cost_sensitive_cross_val_scores_1.4.xlsx')
        best_clf_file_path = os.path.join(
            RESULTS_DIR, clf_type + '_best_clf_cost_sensitive_accuracy_scores_1.4.xlsx')

        # Write results to the EXCEL files
        try:
            if ~naive_clf_scores.empty:
                naive_clf_scores.to_excel(naive_clf_file_path)
            if ~cross_val_scores.empty:
                cross_val_scores.to_excel(best_cross_val_file_path)
            if ~best_clf_scores.empty:
                best_clf_scores.to_excel(best_clf_file_path)
        except Exception as e:
            print('Error: ', str(e))
            print('Problem occured while writing to the EXCEL file.')
            raise


predictor = TruthfulnessCostSensitivePredictor()
