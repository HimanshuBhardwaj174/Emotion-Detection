
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import precision_score, recall_score, roc_auc_score

train_data = pd.read_csv('./data/processed/train_feature.csv')
test_data = pd.read_csv('./data/processed/test_feature.csv')

import pickle
with open('./model/xg.pkl','rb') as f:
    xgb_model = pickle.load(f)

X_test_bow = test_data.drop(columns=['label']).values
y_test = test_data['label']

y_pred = xgb_model.predict(X_test_bow)
y_pred_proba = xgb_model.predict_proba(X_test_bow)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
#classification_rep = classification_report(y_test, y_pred)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred) 
auc = roc_auc_score(y_test, y_pred_proba)

dic ={
    'accuracy': accuracy,
    'precision':precision,
    'recall':recall,
    'auc':auc
}

import json

with open('./reports/metrics.json','w') as f:
    json.dump(dic,f,indent=4)