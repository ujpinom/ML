import numpy as np;import seaborn as sns;import matplotlib.pyplot as plt

class curves():

    def plot_validation_curve(train_score,test_score,params,nombre_parametro=None):

        train_score_m=np.nanmean(train_score,axis=1)
        test_score_m=np.nanmean(test_score,axis=1)
        train_score_std = np.nanstd(train_score, axis=1)
        test_score_std = np.nanstd(test_score, axis=1)

        plt.plot(params,train_score_m,color='blue',marker='x',label='Training accuracy')
        plt.fill_between(params,train_score_m+train_score_std,train_score_m-train_score_std,
                         alpha=0.3,color='blue')
        plt.plot(params, test_score_m, color='red', marker='o', label='Validation accuracy')
        plt.fill_between(params, test_score_m + test_score_std, test_score_m - test_score_std,
                         alpha=0.3, color='red')

        if nombre_parametro:
            plt.xlabel(nombre_parametro)
        plt.ylim(0.8, 1)
        plt.xscale('log')
        plt.ylabel('Precisión')
        plt.title('Validation Curve')
        plt.legend()



