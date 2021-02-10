from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np


class esemble_models(BaseEstimator,ClassifierMixin):


    def __init__(self,estimadores,vote='etiquetas',weights=None):
        self.vote=vote
        self.weights=weights
        self.estimadores=estimadores
        self.named_classifiers={key:value for key,value in _name_estimators(estimadores)}

    def fit(self,X,y):

        self.encoder_=LabelEncoder()
        self.encoder_.fit(y)
        self.clases_=self.encoder_.classes_
        self.array_estimadores=[]

        for estimador in self.estimadores:
            estimador_fit=clone(estimador).fit(X,self.encoder_.transform(y))
            self.array_estimadores.append(estimador_fit)

        return self

    def predict(self,X):

        if self.vote=='etiquetas':
            votos=np.array([i.predict(X) for i in self.array_estimadores]).T

            self.maj_votos_=np.apply_along_axis(lambda x:np.nanargmax(np.bincount(x,weights=self.weights))
                                          ,axis=1,arr=votos)
        else:
            self.maj_votos_=np.nanargmax(self.predict_proba(X),axis=1)



        return self.encoder_.inverse_transform(self.maj_votos_)

    def predict_proba(self,X):

        array_proba=np.array([estimador.predict_proba(X) for estimador in self.array_estimadores])
        maj_votos=np.average(array_proba,axis=0,weights=self.weights)
        return maj_votos
