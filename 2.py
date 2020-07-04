from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import array
from keras.models import load_model

def get_train():
    seq=[[0.0,0.1],[0.1,0.2],[0.2,0.3],[0.3,0.4],[0.4,0.5]]

    seq=array(seq)  #combierte una lista a un numpy array.

    X, y = seq[:,0] , seq[:,1]

    X = X.reshape((len(X),1,1))

    return X,y
