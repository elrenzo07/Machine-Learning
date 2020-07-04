from numpy import array
seq=[[0.0,0.1],[0.1,0.2],[0.2,0.3],[0.3,0.4],[0.4,0.5]]

seq=array(seq)  #combierte una lista a un numpy array.

X, y = seq[:,0] , seq[:,1]
print(X.shape)
X = X.reshape((len(X),1,1))
print(X)

#AHORA X E Y ESTAN EN CONDICIONES DE SER USADOS PARA ENTRENAR LA RED

    

