import numpy as np;
import matplotlib.pyplot as plt;import seaborn as sns;sns.set()
from sklearn.linear_model import LogisticRegression

def weights_vs_C(X,y,random_state=0,penalty='l1'):
    pesos,parametros_C=[],[]
    colores=['blue','orange','green','red','purple','brown','pink','gray','olive','cyan','bisque','lime','gold']

    C=np.arange(-4,6)

    for c in C:
        modelo=LogisticRegression(penalty=penalty,C=10.**c,random_state=random_state,solver='liblinear')
        modelo.fit(X,y)
        pesos.append(modelo.coef_[1])
        parametros_C.append(10.**c)

    pesos = np.array(pesos)

    for index,color in zip(np.arange(pesos.shape[1]),colores):
        plt.plot(parametros_C,pesos[:,index],color=color)

    plt.ylabel('Peso')
    plt.xlabel('C')
    plt.xscale('log')
    plt.xlim(10**(-5),10**5)


