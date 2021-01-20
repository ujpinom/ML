class LGGD():

    import numpy as np;
    def __init__(self,epoch,n_rate,random_state):
        self.epoch=epoch
        self.n_rate=n_rate
        self.random_state=random_state

    def fit(self,X,y):
        import numpy as np;
        rgen=  np.random.RandomState(self.random_state)
        self.pesos=rgen.normal(loc=0,scale=0.01,size=X.shape[1]+1)

        self.costo=[]

        for i in range(self.epoch):
            net=self.net_input(X)
            salida=self.activacion(net)
            errores=y-salida
            self.pesos[1:]+=self.n_rate*X.T.dot(errores)
            self.pesos[0]+=np.sum(errores)*self.n_rate
            costo = (-y.dot(np.log(salida)) -
                    ((1 - y).dot(np.log(1 - salida))));
            self.costo.append(costo)

        return self


    def activacion(self,z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def net_input(self,x):
        return np.dot(x,self.pesos[1:])+self.pesos[0]

    def prefict(self,X):
        return np.where(self.net_input(X)>=0,1,0)
