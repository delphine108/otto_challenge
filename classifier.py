from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
import numpy as np

#import sys
#sys.path.append('C:\\src\c++\\xgboost\\wrapper')
import xgboost as xgb

class Classifier(BaseEstimator):
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.clf1 = None
        self.clf2 = None
        self.param = {'eval_metric':'mlogloss'}
        self.param['num_class'] = 9
        self.param['subsample'] = 0.9        
        self.num_round = 120 
        self.obj1 = 'multi:softprob'
        self.obj2 = 'multi:softmax'
 
    def fit(self, X, y):        
        X = self.scaler.fit_transform(X.astype(np.float32))              
        y = self.label_encoder.fit_transform(y).astype(np.int32)
        dtrain = xgb.DMatrix( X, label=y.astype(np.float32))
        
        self.param['objective'] = self.obj1  
        self.clf1 = xgb.train(self.param, dtrain, self.num_round)  

        self.param['objective'] = self.obj2  
        self.clf2 = xgb.train(self.param, dtrain, self.num_round)
 
    def predict(self, X):
        X = self.scaler.fit_transform(X.astype(np.float32))
        dtest = xgb.DMatrix(X)       
        return self.clf2.predict(dtest)
 
    def predict_proba(self, X):
        X = self.scaler.fit_transform(X.astype(np.float32))
        dtest = xgb.DMatrix(X)
        return self.clf1.predict(dtest)