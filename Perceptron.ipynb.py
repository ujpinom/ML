import numpy as np;

class Perceptron():

    def __init__(self,learning_rate=0.01,n_iter=50,random_state=1):
        self.learning_rate=learning_rate
        self.n_iter=n_iter
        self.rst=random_state;


    def fit(self,X,y):
        """X es un matriz con dimesiones [n_muestras n_caracteristicas]
        y es el vector de etiquetas o vector objetio, donde y es [n_muestras,]

        """
        gen=np.random.RandomState(self.rst)
        self.weights_=np.random.normal(loc=0,scale=0.01,size=1+X.shape[1])

        self.errores_=[]

        for _  in range(self.n_iter):
            errores=0
            for muestra,target in zip(X,y):
                update=self.learning_rate*(target.self.predict(muestra))
                self.weights_[1:]+=muestra*update
                self.weights_[0]+=update
                errores+=int(update!=0)
            self.errores_.append(errores)
        return  self


    def net_input(self,X):
        return np.dot(X,self.weights_[1:]+self.weights_[0])


    def predict(self,muestra):
        return np.where(self.net_input(muestra)>=0,1,-1)

