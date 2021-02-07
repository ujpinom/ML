import seaborn as sns;import matplotlib.pyplot as plt
import numpy as np

class curves(object):

    def plot_learning_curve(training_score,test_score,size):

        training_score_m=np.nanmean(training_score,axis=1)
        test_score_m=np.nanmean(test_score,axis=1)
        training_score_std = np.nanstd(training_score,axis=1)
        test_score_std = np.nanstd(test_score,axis=1)

        plt.style.use('fivethirtyeight')
        plt.plot(size,training_score_m,color='red',marker='o',label='Trainig accuracy')
        plt.fill_between(size,training_score_m+training_score_std,training_score_m-training_score_std,
                         color='red',alpha=0.3)
        plt.plot(size,test_score_m,color='blue',marker='x',label='Validation accuracy')
        plt.fill_between(size, test_score_m + test_score_std, test_score_m - test_score_std,
                         color='blue', alpha=0.3)
        plt.xlabel('Tamaño del conjunto de entrenamiento')
        plt.ylabel('Presición')
        plt.title('Learning Curve')
        plt.legend()
        plt.ylim(0.8, 1.01)