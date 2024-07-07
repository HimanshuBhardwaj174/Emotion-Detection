
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

train_data = pd.read_csv('./data/processed/train_feature.csv')
test_data = pd.read_csv('./data/processed/test_feature.csv')

X_train_bow = train_data.drop(columns=['label']).values
X_test_bow = test_data.drop(columns=['label']).values

y_train = train_data['label']
y_test = test_data['label']




# Define and train the XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train_bow, y_train)



import pickle
import os
os.makedirs('model')
with open('./model/xg.pkl','wb') as f:
    pickle.dump(xgb_model,f)
