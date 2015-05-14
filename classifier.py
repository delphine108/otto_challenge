from sklearn.base import BaseEstimator
 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import os
os.environ["OMP_NUM_THREADS"] = "1"
 
import theano
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.nonlinearities import rectify
from lasagne.updates import nesterov_momentum
from lasagne.updates import adagrad
from nolearn.lasagne import NeuralNet
 
 
class Classifier(BaseEstimator):
 
    def __init__(self):
        self.net = None
        self.label_encoder = None
 
    def fit(self, X, y):
        layers0 = [('input', InputLayer),
           ('dropoutf', DropoutLayer),
           ('dense0', DenseLayer),           
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense2', DenseLayer),           
           ('dropout2', DropoutLayer),
           ('dense3', DenseLayer),
           ('dropout3', DropoutLayer),
           ('output', DenseLayer)]
           
        X = X.astype(theano.config.floatX)
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y).astype(np.int32)
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        num_classes = len(self.label_encoder.classes_)
        num_features = X.shape[1]
        self.net = NeuralNet(layers=layers0,
                             input_shape=(None, num_features),
                             
                             dropoutf_p=0.15,
                             dense0_num_units=1024,                 
                             dropout0_p=0.5,
                             dense0_nonlinearity=rectify,
                             dense1_num_units=1024,
                             dropout1_p=0.15,
                             dense1_nonlinearity=rectify,
                             dense2_num_units=1024,
                             dropout2_p=0.15,
                             dense2_nonlinearity=rectify,
                             dense3_num_units=1024,
                             dropout3_p=0.15,
                             dense3_nonlinearity=rectify,
 
                             update=adagrad,
                             update_learning_rate=0.01,
 
                             eval_size=0.2,
                             verbose=1,
                             max_epochs=150
                             )
        self.net.fit(X, y)
        return self
 
    def predict(self, X):
        X = X.astype(theano.config.floatX)
        X = self.scaler.fit_transform(X)
        return self.label_encoder.inverse_transform(self.net.predict(X))
 
    def predict_proba(self, X):
        X = X.astype(theano.config.floatX)
        X = self.scaler.fit_transform(X)
        return self.net.predict_proba(X)