# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:38:15 2020

@author: Renzo
"""


import numpy as np
import matplotlib.pyplot as plt

class perceptron:
    def __init__(self, n):
        self.pesos = np.random.randn(n)
        self.n=n
    def propagacion(self, entradas):
        self.salida = 1*(np.dot(entradas,self.pesos)>0) #suma ponderada de los pesos por las entradas.
        self.entradas = entradas
    def actualizacion(self, alfa, salidad):
        for i in range(0,self.n):
            self.pesos[i] = self.pesos[i] + alfa*(salidad - self.salida)*self.entradas[i]

perceptron_tres_entradas = perceptron(3)
#verificar si inicio la clase
print(perceptron_tres_entradas.pesos) #si da valores en la consola ==> si inició

perceptron_tres_entradas.propagacion([0, 0, 1])
print(perceptron_tres_entradas.salida) #tiene que devolver 0 o 1

"""COMPUERTA AND - PERCEPTRON"""
AND = perceptron(3)
ejemplos = np.array([[0,0,1,0],[0,1,1,0],[1,0,1,0],[1,1,1,1],[0,1,0,0],[1,0,0,0],[1,1,0,0]])
gradiente = [AND.pesos]
for epoch in range(0,100):
    for i in range(0,len(ejemplos)):
        AND.propagacion(ejemplos[i,0:3])
        AND.actualizacion(0.01, ejemplos[i,3])
        gradiente = np.concatenate((gradiente, [AND.pesos]),axis = 0)
        

"""REPRESENTACIÓN GRÁFICA"""
plt.plot(gradiente[:,0],'k')
plt.plot(gradiente[:,1],'b')
plt.plot(gradiente[:,2],'r')


"""PRUEBA DE FUNCIONAMIENTO"""

AND.propagacion([0,1,1])
print(AND.salida)







