# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 23:26:42 2020

@author: Renzo
"""


import perceptron_1 as p
import numpy as np

in_and = [[0,1],[1,0],[1,1]]
IN_and = np.array(in_and)
OUT_and = np.array([[0],[0],[1]])


_and = p.perceptron()
_and.fit(IN_and,OUT_and)

print(_and.predict([1,1]))
print(_and.predict([1,0]))
print(_and.predict([0,1]))
#print(_and.predict([0,0]))

print(_and.weights)
print(_and.bias)
