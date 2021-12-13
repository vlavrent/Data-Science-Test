import pandas as pd
from sklearn.decomposition import PCA
from Plots import All_plots

class EDA():
    def __init__(self,path):
        self.path = path
        self.target = 'has_pod'
        self.data = self.read_data()
    
    def read_data(self):
        # Read csv file
        return pd.read_csv(self.path)

    def Info(self):
        # Print data info
        print('Data Info'.center(55, '-'))
        print(self.data.info())

    def Binarize(self):
        # Select home_network_type Object type column
        col = self.data.select_dtypes('O').columns[1]

        # Find unique column elements
        uniq = self.data[col].unique()

        # Binarize elements
        self.data[col] = self.data[col].apply(lambda x: 0 if x==uniq[0] else 1)

    def Describe(self):
        # Print data count, mean, standard deviation, min, max
        print('Describe Data'.center(200, '-'))
        print(self.data.describe().to_string)

    def Missing_values(self):
        # Print missing values
        print('Missing Values'.center(30, '-'))
        print(self.data.isnull().sum())


    def Duplicates(self):
        # Print Duplicates
        print("The Number of Duplicates is: "+str(self.data.duplicated().sum()))

        # Drop Duplicates if they exist
        if self.data.duplicated().sum()!=0:
            self.data = self.data.drop_duplicates()

    def Correlation(self):
        # Correlation of columns in respect to target column has_pod
        corr_matrix = self.data.corrwith(self.data[self.target],method='pearson').sort_values(ascending=False)

        # Remove columns with correlation higher than 0.8
        cm = pd.DataFrame(corr_matrix).reset_index()
        remove_columns = list(cm.loc[(cm[0] > 0.8) & (cm['index'] != self.target)]['index'])
        self.data = self.data.drop(columns = remove_columns)

        return corr_matrix

    def pca_outliers(self,pca,upper_lim):
        # Convert array to dataframe
        pca = pd.DataFrame(pca)

        # Find row mean
        pca['mean'] = pca.apply(lambda x: x.mean(), axis=1)

        # Assign color according to upper limit
        pca['color'] = pca.apply(lambda x: '#DC143C' if x['mean'] > upper_lim else '#82C7A5', axis=1)

        return pca

    def Outlier(self):
        # PCA or dimensionality reduction
        pca = PCA(n_components =2).fit_transform(self.data.drop(columns=[self.target,'gw_mac_address']))

        # Find upper limit using mean and std multiplied by a factor
        upper_lim = pca.mean() + pca.std() * 3

        # Find pca outliers
        pca = self.pca_outliers(pca,upper_lim)

        # Concat mean with data and filter
        self.data['mean'] = pca['mean']
        self.data = self.data[self.data['mean']<=upper_lim]

        return pca

    def Imbalance(self):
        # Return Class value count
        return self.data[self.target].value_counts()

    def execute(self):
        # Print info regarding the data type and Non-Null count
        self.Info()

        # Binarize the home_network_type object type column
        self.Binarize()

        # Describe data
        self.Describe()

        # Missing values
        self.Missing_values()

        # Duplicated data
        self.Duplicates()

        # Correlation
        corr_m = self.Correlation()

        # Data Imbalance
        imb = self.Imbalance()

        # Outlier
        pca = self.Outlier()

        # Plot
        settings = {'corr':corr_m,'imb':imb,'pca':pca,'model':None,'X_test':None,'y_test':None,'report':None}
        pl = All_plots(settings)
        pl.plot_eda()

        return self.data

