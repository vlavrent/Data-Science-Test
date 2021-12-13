import matplotlib.pyplot as plt
import  pandas as pd
from sklearn import metrics
import numpy as np
import seaborn as sns

class All_plots():
    def __init__(self,settings):
        self.correlation = settings['corr']
        self.imbalance = settings['imb']
        self.pca = settings['pca']
        self.model = settings['model']
        self.X_test = settings['X_test']
        self.y_test = settings['y_test']
        self.class_report = settings['report']

    def plot_correlation(self):
        # Clarify subplot position
        plt.subplot(3,1,1)

        # Set plot colors
        colors = pd.DataFrame(self.correlation)
        colors = colors[0].apply(lambda x: '#82C7A5' if float(x)>0 else '#DC143C')

        # Plot Correlation
        self.correlation.plot.bar(color=list(colors))

        # Clarify plot labels and title
        plt.xticks(rotation=45,fontsize=6)
        plt.title('Column Correlation with: has_pod')
        plt.xlabel('Column names')
        plt.ylabel('Correlation')


    def plot_imbalance(self):
        # Clarify Subplot position
        plt.subplot(3,1,2)

        # Plot Imbalance
        self.imbalance.plot.bar(color=['#82C7A5','#DC143C'])

        # Clarify plot labels and title
        plt.title('Class Imbalance')
        plt.xlabel('Class Label')
        plt.ylabel('Number of Samples')


    def plot_outliers(self,ax):
        # Clarify Subplot position
        plt.subplot(3,1,3)

        # Plot Outliers
        plt.scatter(x=np.array(self.pca[0]), y=np.array(self.pca[1]), color=self.pca['color'].values.tolist())

        # Clarify plot labels and title
        plt.title('Outliers with Principal Component Analysis')
        plt.xlabel('X label')
        plt.ylabel('Y label')


    def plot_eda(self):
        # Set figsize
        fig, ax = plt.subplots(figsize=(10, 12))

        # Plot Correlation
        self.plot_correlation()

        # Plot Imbalance
        self.plot_imbalance()

        # Plot Outliers
        self.plot_outliers(ax)

        fig.tight_layout(pad=0.4, w_pad=0.4, h_pad=2)

        plt.show()
    def plot_ROC(self):
        # Plot ROC curve
        metrics.plot_roc_curve(self.model, self.X_test, self.y_test)

        # Clarify plot labels and title
        plt.title('ROC Curve')
        plt.show()

    def plot_classification_report(self):
        # Plot heatmap
        sns.heatmap(pd.DataFrame(self.class_report).iloc[:-1, :].T, annot=True)

        # Clarify plot labels and title
        plt.title('Classification Report heatmap')
        plt.show()


    def plot_model(self):
        # Plot ROC curve
        self.plot_ROC()

        # Plot Classification report
        self.plot_classification_report()






