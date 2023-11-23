
import os
import numpy as np
import pandas as pd
import csv
import random
import json
from sklearn import metrics
from sklearn import svm
from sklearn import cluster
import sklearn

import argparse
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn import tree

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from xgboost import XGBClassifier
import xgboost as xgb
from xgboost import plot_importance

import joblib

save_model_dir = None

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def RF(trainX,trainY,testX,testY):
    rfc = RandomForestClassifier(n_estimators=50, class_weight='balanced')
    rfc.fit(trainX, trainY)
#     importances = rfc.feature_importances_
#     indices = np.argsort(importances)[::-1]
#     fea = list(X.columns[indices[:1000]])
#     imp = list(importances[indices[:1000]])
    
    if save_model_dir != None:
        joblib.dump(rfc, os.path.join(save_model_dir, 'RF.pkl'))

    predict_result = rfc.predict(testX)
    # print(testY[predict_result != testY])
    acc = metrics.accuracy_score(testY,predict_result)
    recall = metrics.recall_score(testY, predict_result, average="macro")
    precision = metrics.precision_score(testY, predict_result, average="macro")
    F1 = metrics.f1_score(testY,predict_result,average="macro") #weighted
    return acc,recall,precision,F1

def DF(trainX,trainY,testX,testY):
    dfc = tree.DecisionTreeClassifier()
    dfc.fit(trainX,trainY)

    if save_model_dir != None:
        joblib.dump(dfc, os.path.join(save_model_dir, 'DF.pkl'))
    
    predict_result = dfc.predict(testX)
    acc = metrics.accuracy_score(testY,predict_result)
    recall = metrics.recall_score(testY, predict_result, average="macro")
    precision = metrics.precision_score(testY, predict_result, average="macro")
    F1 = metrics.f1_score(testY,predict_result,average="macro")
    return acc,recall,precision,F1

def XGB(trainX,trainY,testX,testY):
    dtrain = xgb.DMatrix(trainX, label=trainY)
    
    num_XGB_round = 2
    XGB_params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'gamma': 0.1,
        'max_depth': 8,
        'alpha': 0,
        'lambda': 0,
        'subsample': 0.7,
        'colsample_bytree': 0.5,
        'min_child_weight': 3,
        'silent': 0,
        'eta': 0.03,
        'nthread': -1,
        'seed': 2021,
        'scale_pos_weight': 0.1
#         param['num_class'] = 6
    }
    
    xgb_model = xgb.train(XGB_params, dtrain, num_XGB_round)
#     model_path = 'xgb.model'
#     bst.save_model(model_path)
#     model = xgb.XGBClassifier()
#     model.load_model(model_path)
    
    if save_model_dir != None:
        joblib.dump(xgb_model, os.path.join(save_model_dir, 'XGB.pkl'))

    dtest = xgb.DMatrix(testX)
    predict_result = xgb_model.predict(dtest)
#     predict_contribs = xgb_model.predict(dtest, pred_contribs=True)
#     sum(predict_contribs[0])
    predict_result = (predict_result >= 0.5) * 1
#     print(predict_result)
    acc = metrics.accuracy_score(testY,predict_result)
    recall = metrics.recall_score(testY, predict_result, average="macro")
    precision = metrics.precision_score(testY, predict_result, average="macro")
    F1 = metrics.f1_score(testY,predict_result,average="macro")
    return acc,recall,precision,F1

def SVM(trainX,trainY,testX,testY):
    clf = svm.SVC(probability=True)
    clf.decision_function_shape="ovr"
    clf.fit(trainX,trainY)

    if save_model_dir != None:
        joblib.dump(clf, os.path.join(save_model_dir, 'SVM.pkl'))

    predict_result = clf.predict(testX)
    acc = metrics.accuracy_score(testY,predict_result)
    recall = metrics.recall_score(testY, predict_result, average="macro")
    precision = metrics.precision_score(testY, predict_result, average="macro")
    F1 = metrics.f1_score(testY, predict_result, average="macro")
    return acc,recall,precision,F1

def kmeans(trainX,testX,testY,k=2):
    # print(k)
    model = cluster.KMeans(n_clusters=k)
    model.fit(trainX)

    if save_model_dir != None:
        joblib.dump(model, os.path.join(save_model_dir, 'KMEANS.pkl'))

    predict_result = model.predict(testX)
    acc = metrics.accuracy_score(testY, predict_result)
    recall = metrics.recall_score(testY, predict_result, average="macro")
    precision = metrics.precision_score(testY, predict_result, average="macro")
    F1 = metrics.f1_score(testY, predict_result, average="macro")
    return acc, recall, precision, F1

def DBSCAN(trainX,testX,testY,eps=1,min_samples=2):
    model = cluster.DBSCAN(eps=eps,min_samples=min_samples)
    model.fit(trainX)

    if save_model_dir != None:
        joblib.dump(model, os.path.join(save_model_dir, 'DBSCAN.pkl'))

    predict_result = model.fit_predict(testX)
    predict_result[predict_result == -1] = 1
    acc = metrics.accuracy_score(testY, predict_result)
    recall = metrics.recall_score(testY, predict_result, average="macro")
    precision = metrics.precision_score(testY, predict_result, average="macro")
    F1 = metrics.f1_score(testY, predict_result, average="macro")
    return acc, recall, precision, F1


def RF_model(testX,testY):
    rfc = joblib.load(os.path.join(save_model_dir, 'RF.pkl'))

    predict_result = rfc.predict(testX)
    # print(testY[predict_result != testY])
    acc = metrics.accuracy_score(testY,predict_result)
    recall = metrics.recall_score(testY, predict_result, average="macro")
    precision = metrics.precision_score(testY, predict_result, average="macro")
    F1 = metrics.f1_score(testY,predict_result,average="macro") #weighted
    return acc,recall,precision,F1

def DF_model(testX,testY):
    dfc = joblib.load(os.path.join(save_model_dir, 'DF.pkl'))
    
    predict_result = dfc.predict(testX)
    acc = metrics.accuracy_score(testY,predict_result)
    recall = metrics.recall_score(testY, predict_result, average="macro")
    precision = metrics.precision_score(testY, predict_result, average="macro")
    F1 = metrics.f1_score(testY,predict_result,average="macro")
    return acc,recall,precision,F1

def XGB_model(testX,testY):
    xgb_model = joblib.load(os.path.join(save_model_dir, 'XGB.pkl'))

    dtest = xgb.DMatrix(testX)
    predict_result = xgb_model.predict(dtest)
#     predict_contribs = xgb_model.predict(dtest, pred_contribs=True)
#     sum(predict_contribs[0])
    predict_result = (predict_result >= 0.5) * 1
#     print(predict_result)
    acc = metrics.accuracy_score(testY,predict_result)
    recall = metrics.recall_score(testY, predict_result, average="macro")
    precision = metrics.precision_score(testY, predict_result, average="macro")
    F1 = metrics.f1_score(testY,predict_result,average="macro")
    return acc,recall,precision,F1

def SVM_model(testX,testY):
    clf = joblib.load(os.path.join(save_model_dir, 'SVM.pkl'))

    predict_result = clf.predict(testX)
    acc = metrics.accuracy_score(testY,predict_result)
    recall = metrics.recall_score(testY, predict_result, average="macro")
    precision = metrics.precision_score(testY, predict_result, average="macro")
    F1 = metrics.f1_score(testY, predict_result, average="macro")
    return acc,recall,precision,F1

def kmeans_model(testX,testY,k=2):
    # print(k)
    model = joblib.load(os.path.join(save_model_dir, 'KMEANS.pkl'))

    predict_result = model.predict(testX)
    acc = metrics.accuracy_score(testY, predict_result)
    recall = metrics.recall_score(testY, predict_result, average="macro")
    precision = metrics.precision_score(testY, predict_result, average="macro")
    F1 = metrics.f1_score(testY, predict_result, average="macro")
    return acc, recall, precision, F1

def DBSCAN_model(testX,testY,eps=1,min_samples=2):
    model = joblib.load(os.path.join(save_model_dir, 'DBSCAN.pkl'))

    predict_result = model.fit_predict(testX)
    predict_result[predict_result == -1] = 1
    acc = metrics.accuracy_score(testY, predict_result)
    recall = metrics.recall_score(testY, predict_result, average="macro")
    precision = metrics.precision_score(testY, predict_result, average="macro")
    F1 = metrics.f1_score(testY, predict_result, average="macro")
    return acc, recall, precision, F1


def detect():
    traindata_path = sys.argv[1]
    testdata_path = sys.argv[2]
    model = sys.argv[3]
    mode = sys.argv[4]

    global save_model_dir
    if len(sys.argv) >= 6:
        save_model_dir = sys.argv[5]

    if mode == 'train':
        train_datasets = pd.read_csv(traindata_path)
        test_datasets = pd.read_csv(testdata_path)

        trainX = train_datasets.iloc[:,:106].values #trainX = vectors.iloc[:,:546].values :,1:31
        trainY = train_datasets.iloc[:,106].values.astype('int')

        testX = test_datasets.iloc[:,:106].values
        testY = test_datasets.iloc[:,106].values.astype('int')

        if model == 'RF':
            acc,recall,precision,f1 = RF(trainX,trainY,testX,testY)
            print("acc:", acc, "recall:", recall, "precision:", precision, "f1:", f1)
        elif model == 'DF':
            acc,recall,precision,f1 = DF(trainX,trainY,testX,testY)
            print("acc:", acc, "recall:", recall, "precision:", precision, "f1:", f1)
        elif model == 'XGB':
            acc,recall,precision,f1 = XGB(trainX,trainY,testX,testY)
            print("acc:", acc, "recall:", recall, "precision:", precision, "f1:", f1)
        elif model == 'SVM':
            acc,recall,precision,f1 = SVM(trainX,trainY,testX,testY)
            print("acc:", acc, "recall:", recall, "precision:", precision, "f1:", f1)
        elif model == 'kmeans':
            acc,recall,precision,f1 = kmeans(trainX,testX,testY)
            print("acc:", acc, "recall:", recall, "precision:", precision, "f1:", f1)
        elif model == 'DBSCAN':
            acc,recall,precision,f1 = DBSCAN(trainX,testX,testY)
            print("acc:", acc, "recall:", recall, "precision:", precision, "f1:", f1)
        elif model == 'ALL':
            print('RF:')
            acc,recall,precision,f1 = RF(trainX,trainY,testX,testY)
            print("acc:", acc, "recall:", recall, "precision:", precision, "f1:", f1)
            print('DF:')
            acc,recall,precision,f1 = DF(trainX,trainY,testX,testY)
            print("acc:", acc, "recall:", recall, "precision:", precision, "f1:", f1)
            print('XGB:')
            acc,recall,precision,f1 = XGB(trainX,trainY,testX,testY)
            print("acc:", acc, "recall:", recall, "precision:", precision, "f1:", f1)
            print('SVM:')
            acc,recall,precision,f1 = SVM(trainX,trainY,testX,testY)
            print("acc:", acc, "recall:", recall, "precision:", precision, "f1:", f1)
            print('kmeans:')
            acc,recall,precision,f1 = kmeans(trainX,testX,testY)
            print("acc:", acc, "recall:", recall, "precision:", precision, "f1:", f1)
            print('DBSCAN:')
            acc,recall,precision,f1 = DBSCAN(trainX,testX,testY)
            print("acc:", acc, "recall:", recall, "precision:", precision, "f1:", f1)
    elif mode == 'test':
        vectors = pd.read_csv(test_path)
        testX = vectors.iloc[:,:546].values
        benign_labels = [label_val for label_val in vectors['label'].unique() if 'benign' in label_val]
        vectors.loc[~vectors['label'].isin(benign_labels), 'label'] = 1
        vectors.loc[vectors['label'].isin(benign_labels), 'label'] = 0
        # vectors.loc[vectors['label'] != 'Benign', 'label'] = 1
        # vectors.loc[vectors['label'] == 'Benign', 'label'] = 0
        testY = vectors['label'].values.astype('int')

        if model == 'RF':
            acc,recall,precision,f1 = RF_model(testX,testY)
            print("acc:", acc, "recall:", recall, "precision:", precision, "f1:", f1)
        elif model == 'DF':
            acc,recall,precision,f1 = DF_model(testX,testY)
            print("acc:", acc, "recall:", recall, "precision:", precision, "f1:", f1)
        elif model == 'XGB':
            acc,recall,precision,f1 = XGB_model(testX,testY)
            print("acc:", acc, "recall:", recall, "precision:", precision, "f1:", f1)
        elif model == 'SVM':
            acc,recall,precision,f1 = SVM_model(testX,testY)
            print("acc:", acc, "recall:", recall, "precision:", precision, "f1:", f1)
        elif model == 'kmeans':
            acc,recall,precision,f1 = kmeans_model(testX,testY)
            print("acc:", acc, "recall:", recall, "precision:", precision, "f1:", f1)
        elif model == 'DBSCAN':
            acc,recall,precision,f1 = DBSCAN_model(testX,testY)
            print("acc:", acc, "recall:", recall, "precision:", precision, "f1:", f1)
        elif model == 'ALL':
            print('RF:')
            acc,recall,precision,f1 = RF_model(testX,testY)
            print("acc:", acc, "recall:", recall, "precision:", precision, "f1:", f1)
            print('DF:')
            acc,recall,precision,f1 = DF_model(testX,testY)
            print("acc:", acc, "recall:", recall, "precision:", precision, "f1:", f1)
            print('XGB:')
            acc,recall,precision,f1 = XGB_model(testX,testY)
            print("acc:", acc, "recall:", recall, "precision:", precision, "f1:", f1)
            print('SVM:')
            acc,recall,precision,f1 = SVM_model(testX,testY)
            print("acc:", acc, "recall:", recall, "precision:", precision, "f1:", f1)
            print('kmeans:')
            acc,recall,precision,f1 = kmeans_model(testX,testY)
            print("acc:", acc, "recall:", recall, "precision:", precision, "f1:", f1)
            print('DBSCAN:')
            acc,recall,precision,f1 = DBSCAN_model(testX,testY)
            print("acc:", acc, "recall:", recall, "precision:", precision, "f1:", f1)

if __name__ == "__main__":
    detect()