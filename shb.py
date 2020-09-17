import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error
import pickle
# import statsmodels.api as sm

class MakeModel:
    def __init__(self, data, cat_cols=None, cont_cols=None, target='price'):
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
        self.model = None
        self.target = target
        self.dropped_columns = []

    def current_model_version(self):
        print('='*40)
        print(f'Currently working on model #{len(self.models)+1}')
        print('='*40)

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
        option = input("Choose an option (1) none (2) zscore method (3) IQR method (4) Drop column: ")
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

    def imuter(self,col,option='median'):
        """
        data imuter that replaces missing value by certain value
        ==================
        input parameters:

        col : string. column name that needs to be immuted

        option : {'median', 'mean', 0, str}
            - 'median' will replace missing values by its median
            - 'mean' will replace missing values by its mean
            - 0 will replace missing values by 0
            - str will replace missing values by given string
        ===================
        output:

        pd.Series
        """
        data = self.data[col]
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
        try:
            log_scaler = FunctionTransformer(np.log1p, validate=True)
            log_data = log_scaler.fit_transform(data)
            sns.distplot(log_data, bins='auto', ax=axes[3])
            axes[3].set(title=f'Log Scaler (normaltest={stats.normaltest(log_data)[1]})')
        except:
            print('Error occured')
        
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

        correlation matrix
        """
        if option == "cat":
            data = self.data[self.cat_cols].copy()
            title = 'Categorical Columns Correlation Heatmap'
        elif option == "cont":
            data = self.data[self.cont_cols].copy()
            title = 'Continuous Columns Correlation Heatmap'
        else:
            data = self.data.copy()
            title = 'All Columns Correlation Heatmap'
       
        # find correlation matrix between different variables given in data
        corr = data.corr()
        # creates triangular mask over the map
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        fig, ax = plt.subplots(figsize=(18,12))
        sns.heatmap(corr, annot=annot, cmap=cmap, mask=mask, ax=ax)
        ax.set(title=title)
        plt.show()
        return np.abs(corr * (~mask))

    def multicolinearity(self, option=None):
        """
        Allows the user to examine and get rid of possible multicolinearity variables using (1) correlation map and (2) 
        ===================
        input parameters:
        
        option : {'cat', 'cont', }
            - 'cat' : outputs the correlation heatmap of categorical variables
            - 'cont' : outputs the correlation heatmap of continuous variables
            - None : outputs the correlation heatmap of all variables
        
        =============
        output:

        ???
        """
        # copies data according to option
        if option == "cat":
            data = self.data[self.cat_cols].copy()
        elif option == "cont":
            data = self.data[self.cont_cols].copy()
        else:
            data = self.data.copy()
        
        # prints and creates correlation heatmap/matrix
        cor = self.corr_map(option=option)

        # allows users to pick which columns to get rid of according to their correlation values
        while True:
            print(cor.unstack().sort_values(ascending=False).head(10))
            user_input = input('Write column name you would like to get rid of (Enter "x" to exit): ')
            if user_input in data.columns:
                self.delete_column(cols=user_input)
            elif user_input == 'x':
                print_message(f"Exiting!\nThere are {self.data.shape[1]} columns remaining in the data")
                break
            else: 
                message = f"'{user_input}' is not found. Please try again."
                print_message(message)

    def delete_column(self, cols):
        """
        delete a column
        ==================
        parameters

        col = string or array. name of the column that needs to be deleted

        ==================
        output

        None
        """
        # check if col exists in the data column.
        if type(cols) == str:
            if cols in self.data.columns:
                self.data.drop(columns=[cols],inplace=True)
                self.dropped_columns.append(cols)
                print_message([f"'{cols}' has been DELETED!", f"There are now {self.data.shape[1]} columns in the data."])

                # also updates cat/cont columns
                if cols in self.cat_cols:
                    self.cat_cols.remove(cols)
                elif cols in self.cont_cols:
                    self.cont_cols.remove(cols)
            else:
                message = f"'{cols}' is not found. Please try again."
                print_message(message)
        # elif type(cols) == list:
        #     for col in cols:
        #         if self.data.columns
        else:
            print_message("Invalid input!")

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
        data = ['X_train', 'X_test', 'y_train', 'y_test']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=train_size, shuffle=shuffle, random_state=random_state)
        # messages = [f'Shape of {x}: {exec("self."+f"x"+".shape")}' for x in data]
        
        messages = [f'Shape of X_train: {self.X_train.shape}', f'Shape of y_train: {self.y_train.shape}', f'Shape of y_train: {self.y_train.shape}', f'Shape of y_test: {self.y_test.shape}']
        print_message(messages)

        # print(f'Shape of y_train: {self.y_train.shape}')
        # print(f'Shape of X_test: {self.X_test.shape}')
        # print(f'Shape of y_test: {self.y_test.shape}')
        # # print('='*40)

    def get_formula(self, cat=True, cont=True):
        """
        creates formula that can be used in OLS regression.
        ====================
        parameters

        cat : boolean. Categorical columns will not be included in the formula if False. Default = True

        cont : boolean. Continuous columns will not be included in the formula if False.Default = True
        =====================
        output

        formula : string.
        """
        # initializes empty lists for features in cont and cat columns
        features_cont = []
        features_cat = []

        # Fills in cat/cont lists according to given paremeters
        if cat:
            features_cat = self.cat_cols
        if cont:
            features_cont = self.cont_cols
        features = features_cat + features_cont

        features = " + ".join(features)
        
        self.target = 'price' # temporary
        formula = self.target + ' ~ ' + features

        return formula

    def regression(self, formula):
        # initialize self.model
        data = pd.concat([self.X_train, self.y_train], axis=1)
        self.model = smf.ols(formula=formula, data=data).fit()
        self.note = None

        return self.model


    def validate_model(self):
        self.y_hat_train = self.model.predict(self.X_train) 
        self.y_hat_test = self.model.predict(self.X_test)

        self.resid_train = self.y_hat_train - self.y_train 
        self.resid_test = self.y_hat_test - self.y_test
        
        train_mse = mean_squared_error(self.y_train)


        ## QQ plot

        ## Homo...
        fig, axs = plt.subplots(ncols=2, figsize=(18,12))
        sns.scatterplot(y_train, self.resid_train, ax=axs[0], label='Training Residuals')
        axs[0].set(title='Training Residual Graph', xlabel='y_train', ylabel='Residuals')
        axs[0].axhline(0, label='zero')

        sns.scatterplot(y_test, self.resid_test, ax=axs[1], label='Testing Residuals')
        axs[1].set(title='Test Residual Graph', xlabel='y_test', ylabel='Residuals')
        axs[1].axhline(0, label='zero')

    def qqplot(self):
        pass

    # def load_model(self):
    #     """
    #     Loads model that has been previously worked on.
    #     ====================
    #     parameters
    #     ====================
    #     output
    #     model
    #     """
    #     if len(self.models) == 0:
    #         print('There are no models saved currently.')
    #     else:
    #         print('Models saved:')
    #         for idx, model in enumerate(self.models):
    #             print(f'({idx+1}) Model #{idx+1}')
    #         print('')
    #         while True:
    #             option = input('Which model would you like to load (enter 0 to exit): ')
    #             if int(option) in range(1,len(self.models)+1):
    #                 self.model = self.models[option-1]
    #                 break
    #             elif int(option) == 0:
    #                 break
    #             else:
    #                 print('invalid option. Please try again.')


    
def print_message(messages, marker="=", number=40):
    """
    prints out messages
    ===================
    parameters

    messages : string or array. 

    marker : string. default = "="

    number : int. the number of marker this method will print out. default = 40
    """
    print(marker*number)
    if type(messages) == str:
        print(messages)
    else:
        for message in messages:
            print(message)
    print(marker*number)

def save_data(filename='model_file', data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_data(filename='model_file'):
    with open(filaname, 'rb') as f:
        return pickle.load(f)