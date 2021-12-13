from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from Plots import All_plots
import seaborn as sns
import pandas as pd

class Liberty_Model():
    def __init__(self,data,model):
        self.model = model
        self.data = data
        self.X = self.data.drop(columns=['has_pod','gw_mac_address'])
        self.y = self.data['has_pod']

    def balanced_classifier_RF(self,X_train, X_test, y_train, y_test):
        # Create Pipeline steps
        steps = [('scaler', MinMaxScaler()), ('model', BalancedRandomForestClassifier(random_state=42, max_depth=5))]

        # Fit data to Pipeline
        pipeline = Pipeline(steps=steps).fit(X_train, y_train)

        return pipeline, classification_report(y_test, pipeline.predict(X_test),output_dict=True)

    def balanced_classifier_LR(self,X_train, X_test, y_train, y_test):
        # Create Pipeline steps
        steps = [('scaler', MinMaxScaler()), ('o', SMOTE(sampling_strategy=0.7)),
                 ('u', RandomUnderSampler(sampling_strategy=0.8)), ('model', LogisticRegression(penalty='l2'))]

        # Fit data to Pipeline
        pipeline = Pipeline(steps=steps).fit(X_train, y_train)

        return pipeline, classification_report(y_test, pipeline.predict(X_test),output_dict=True)

    def lmodel(self):
        # Train Test split
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.30,random_state=25)

        # Model selection
        if self.model=='RF':
            model,class_report = self.balanced_classifier_RF(X_train, X_test, y_train, y_test)
        elif self.model=='LR':
            model, class_report = self.balanced_classifier_LR(X_train, X_test, y_train, y_test)


        settings = {'corr': None, 'imb': None, 'pca': None, 'model': model, 'X_test': X_test, 'y_test': y_test,'report':class_report}
        pl = All_plots(settings)
        pl.plot_model()








