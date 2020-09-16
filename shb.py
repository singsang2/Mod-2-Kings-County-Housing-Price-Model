import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
# import statsmodels.api as sm

class MakeModel:
    def __init__(self, data, cat_cols=None, cont_cols=None):
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.data = data
        self.org_size = len(data)
        self.current_size = self.get_current_size()
        self.percent_data = 100
        self.transformers = {} # scalers
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def get_current_size(self):
        """
        returns the current data size in the model.
        """
        return len(self.data)

    def outlier(self, col):
        """
        Eliminates outliers using different methods
        =================
        parameters:
        
        col : column name

        option : {'zscore', 'IQR'}
            - zscore : anything above abs(z-score) of 3 will be eliminated
            - IQR : anything beyond 1.5 of IQR will be eliminated
        
        verbose : Default False.
            if True it will print how many data points were eliminated
        ===================
        output:

        new pd.Series
        """

        fig, axes = plt.subplots(ncols=3, figsize=(18,12))
        
        data =self.data[col].copy()
        sns.distplot(data, bins='auto', ax=axes[0], label=f'Original (n={len(data)})')
        axes[0].set(title=f'Original (n={len(data)})')

        zscore_mask = self.zscore_outlier(data)
        data_zscore = data[~zscore_mask]
        sns.distplot(data_zscore, bins='auto', ax=axes[1], label=f'Zscore Outlier (n={len(data_zscore)}, {len(data_zscore)/len(data)*100}%)')
        axes[1].set(title=f'Zscore Outlier (n={len(data_zscore)}, {len(data_zscore)/len(data)*100}%)')

        iqr_mask = self.IQR_outlier(data)
        data_iqr = data[~iqr_mask]
        sns.distplot(data_iqr, bins='auto', ax=axes[2], label=f'IQR Outlier (n={len(data_iqr)}, {len(data_iqr)/len(data)*100}%)')
        axes[2].set(title=f'IQR Outlier (n={len(data_iqr)}, {len(data_iqr)/len(data)*100}%)')

        plt.show()
        option = input("Choose an option (1) none (2) zscore method (3) IQR method: ")
        if option == '1':
            print('nothing has changed.')
        elif option == '2':
            self.update_data(self.data[~zscore_mask])
        elif option == '3':
            self.update_data(self.data[~iqr_mask])
        else:
            print('nothing has changed.')

    def update_data(self, data):
        before = self.current_size
        self.data = data
        self.current_size = self.get_current_size()
        self.percent_data = round(self.current_size/self.org_size*100,3)
        print('\n')
        print('='*40)
        print(f'{before - self.current_size} number of data have been removed by this process.\n')
        print(f'So far we have {self.percent_data}% of original data.')
        print('='*40)

    def zscore(self, data):
        """
        Converts pd.Series into zscores
        =================
        parameters:

        data = pd.Series

        =================
        Output:

        pd.Series of zscores
        """
        return (data-data.mean())/data.std()

    def zscore_outlier(self, data, score=3):
        """
        filters out data that are beyond certain zscores.
        ==================
        parameters

        data : pd.Series

        score=3 : cut off zscore. Default to 3.
        ==================
        output

        numpy array with booleans values that need to be filtered out.
        """
        zscore_data = self.zscore(data)
        mask = np.abs(zscore_data) > 3
        return mask

    def IQR_outlier(self, data, factor=1.5):
        """
        Filters out data that are beyond 1.5 IQR from the Q1 or Q3.
        ====================
        parameters

        data : pd.Series

        factor=1.5 : factor you would like to multiply IQR by. Defaults to 1.5
        ====================
        output

        numpy array with booleans values that need to be filtered out.
        """
        iqr = stats.iqr(data)
        median = data.median()
        q1 = np.percentile(data, 25) # Q1
        q3 = np.percentile(data, 75) # Q3
        mask = (data > q3 + 1.5*iqr) | (data < q1 - 1.5*iqr)
        return mask

    def imuter(data,option='median'):
        """
        data imuter that replaces missing value by certain value
        ==================
        input parameters:

        data = pd.Series and it could be either categorical or numerical

        option = {'median', 'mean', 0, str}
            - 'median' will replace missing values by its median
            - 'mean' will replace missing values by its mean
            - 0 will replace missing values by 0
            - str will replace missing values by given string
        ===================
        output:

        pd.Series
        """
        if option == 'median':
            data.fillna(data.median())
        elif option == 'mean':
            data.fillna(data.mean())
        elif option == 0:
            data.fillna(0)
        else:
            data.fillna(option)
        
        return data

    def scaler(self, col):
        """
        Allows the user to choose which scaler to be used on selected 'col' of the data.
        ===================
        input parameters:
        
        col = name of the column you would like to work with
        ===================
        output

        scaled pd.Series version.
        """ 
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18,12))
        
        axes = axes.ravel() # flattens matrix
        
        # prints the histogram of the original data
        data =self.data[[col]].copy()
        sns.distplot(data, bins='auto', ax=axes[0], label=f'Original (n={len(data)})')
        axes[0].set(title=f'Original (normaltest={stats.normaltest(data)[1]})')

        # for transformer in transfomers:

        s_scaler = StandardScaler()
        s_data = s_scaler.fit_transform(data)
        sns.distplot(s_data, bins='auto', ax=axes[1])
        axes[1].set(title=f'Standard Scalers (normaltest={stats.normaltest(s_data)[1]})')

        mm_scaler = MinMaxScaler()
        mm_data = mm_scaler.fit_transform(data)
        sns.distplot(mm_data, bins='auto', ax=axes[2])
        axes[2].set(title=f'MinMax Scaler (normaltest={stats.normaltest(mm_data)[1]})')

        log_scaler = FunctionTransformer(np.log1p, validate=True)
        log_data = log_scaler.fit_transform(data)
        sns.distplot(log_data, bins='auto', ax=axes[3])
        axes[3].set(title=f'Log Scaler (normaltest={stats.normaltest(log_data)[1]})')
        
        plt.suptitle(f'Different scalers for {col} column (n={len(data)})')
        plt.show()
        option = input("Choose an option (1) none (2) standard (3) min_max (4) logarithmic: ")
        if option == '1':
            print('Nothing has changed.')
        elif option == '2':
            self.data[col] = s_data
            self.transformers[col] = s_scaler
        elif option == '3':
            self.data[col] = mm_data
            self.transformers[col] = mm_scaler
        elif option == '4':
            self.data[col] = log_data
            self.transformers[col] = log_scaler
        else:
            print('Nothing has changed.')

    def corr_map(self, option=None, annot=True, cmap='YlGnBu'):
        """
        Creates and prints a heatmap of correlation values between given data.
        ===================
        input parameters:
        
        option : {'cat', 'cont', }
            - 'cat' : outputs the correlation heatmap of categorical variables
            - 'cont' : outputs the correlation heatmap of continuous variables
            - None : outputs the correlation heatmap of all variables
        annot : True by default. Can decide whether you would like to see the values on the heatmap graph.

        cmap : 'YlGnBu' by default. seaborn colormap themes
        =============
        output:

        heatmap of correlation matrix
        """
        if option == "cat":
            data = self.data[self.cat_cols].copy()
        elif option == "cont":
            data = self.data[self.cont_cols].copy()
        else:
            data = self.data.copy()
       
        # find correlation matrix between different variables given in data
        corr = data.corr()
        # creates triangular mask over the map
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        fig, ax = plt.subplots(figsize=(18,12))
        sns.heatmap(corr, annot=annot, cmap=cmap, mask=mask, ax=ax)
        return fig

    def split(self, train_size=0.75, shuffle=True, random_state=42):
        """
        splits the data into test and train data set using sklearn.model_selection
        ===================
        parameters

        train_size : float. Determines how much percent of original data you would want as train set. (Default = 0.75)

        shuffle : boolean. (Default = True)

        random_state : int. random state (Default = 42)
        ===================
        output

        X_train, X_test, y_train, y_test
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=train_size, shuffle=shuffle, random_state=random_state)
        print('='*40)
        print(f'Shape of X_train: {self.X_train.shape}')
        print(f'Shape of y_train: {self.y_train.shape}')
        print(f'Shape of X_test: {self.X_test.shape}')
        print(f'Shape of y_test: {self.y_test.shape}')
        print('='*40)

    def regression(self):
        pass

    