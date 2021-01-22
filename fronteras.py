
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class visualizador(object):

    def visualizar_fronteras(X, y, modelo, muestreo=0.02, test_set=None):

        marcadores = ['^', 'v', 'o', '^', 'v']
        colores = ['red', 'yellow', 'black', 'yellow', 'gray']

        mapa_colores = ListedColormap(colores[:len(np.unique(y))])

        x_min_f1, x_max_f1 = (X[:, 0]).min() - 1, (X[:, 0]).max() + 1  ##Característica Longitud_cépalo
        x_min_f2, x_max_f2 = (X[:, 1]).min() - 1, (X[:, 1]).max() + 1  ###Característicaa longitud_pétalo

        vector1 = np.arange(x_min_f1, x_max_f1, muestreo)
        vector2 = np.arange(x_min_f2, x_max_f2, muestreo)

        x_meshgrid, y_meshgrid = np.meshgrid(vector1, vector2)

        Z = modelo.predict(np.array([x_meshgrid.ravel(), y_meshgrid.ravel()]).T)
        Z = Z.reshape(x_meshgrid.shape)
        plt.contourf(x_meshgrid, y_meshgrid, Z, alpha=0.2, cmap=mapa_colores)

        plt.xlim(x_min_f1, x_max_f1)
        plt.ylim(x_min_f2, x_max_f2)

        for i, index in enumerate(np.unique(y)):
            plt.scatter(x=X[y == index, 0], y=X[y == index, 1], marker=marcadores[i], alpha=0.8, edgecolor='black',
                        label=index)
            plt.legend(loc='lower right')
        if test_set:
            x = X[test_set, :]
            plt.scatter(x[:, 0], x[:, 1], c='', edgecolor='black', alpha=1.0, linewidth=1, marker='o',
                        s=100, label='conjunto_testeo')