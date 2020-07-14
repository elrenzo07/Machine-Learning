# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 17:18:11 2020

@author: Renzo
"""

import numpy as np

class perceptron:
    def __init__(self, lr = 0.01, n_iters=100):
        self.lr=lr
        self.n_iters = n_iters
        self.activation = self._step_funct
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_rows,n_columns = X.shape
        #inicializar pesos
        self.weights=np.zeros(n_columns)
        self.bias = 0
        
        y_ = np.array([1 if i > 0 else 0 for i in y]) #funcion escalon
        
        for _ in range(self.n_iters):
            #numero de epocas de entrenamiento
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation(linear_output)
                
                #Actualización de los pesos
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
                
    def predict(self, X):
        linear_output = np.dot(X,self.weights) + self.bias
        y_predicted = self.activation(linear_output)
        return y_predicted
        
        
    def _step_funct(self, x):
        return np.where(x>=0,1,0)