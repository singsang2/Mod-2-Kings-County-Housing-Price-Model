import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib as mpl
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pickle
import timeit
# import statsmodels.api as sm
plt.style.use('seaborn-whitegrid')
mpl.rcParams["figure.titlesize"] = 12

class MakeModel:
    def __init__(self, data, cat_cols=[], cont_cols=[], target='price'):
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.data = data.copy()
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
        self.splitted = False
        self.ohe_dropped_columns = []
        self.ohe_cols = []
        self.train_mse = None
        self.test_mse = None
        self.train_r2 = None
        self.test_r2 = None

    def __author__(self):
        message = "Sung Hoon Bae"
        print_message(message)
        # return message
    
    def __str__(self):
        print("Model Information")
        messages = [
                    f"Shape:\t{self.data.shape}",
                    f"Perecent Retained:\t{self.percent_data}",
                    f"Dropped columns:\t{self.dropped_columns}",
                    f"Categorical columns:\t{self.cat_cols}",
                    f"Contiuous columns:\t{self.cont_cols}",
                    f"Number of OHE columns:\t{len(self.ohe_cols)}",
                    f'Train MSE = {self.train_mse}\tTrain R2 = {self.train_r2}', 
                    f'Test MSE = {self.test_mse}\tTest R2 = {self.test_r2}'
                   ]
        print_message(messages)
        return "Model Information Completed"

    def __repr__(self):
        return self.__str__()

    def col_classifier(self):
        """
        Allows users to classify columns into either categorical or continuous by examining linearity of each columns.
        """
        # Initialize lists
        self.cat_cols = []
        self.cont_cols = []

        # Determining categorical and continous columns by examining histograms
        for col in self.data.columns:
            fig, ax = plt.subplots(figsize=(10,7))
            try:
                sns.regplot(x=col, y=self.target, data=self.data, ax=ax)
                ax.set(title=f'Linearity of column {col}', xlabel=f'{col}', ylabel=self.target)
                plt.show()
                print("""
                      Options
                      1. Categorical column
                      2. Continuous column
                      3. Drop column
                      """)
                user_input = input('Classify the column: ')
                if user_input=='1':
                    self.cat_cols.append(col)
                    message = f"'{col}' has been added to categorical columns!'"
                    print_message(message)
                elif user_input=='2':
                    self.cont_cols.append(col)
                    message = f"'{col}' has been added to continuous columns!'"
                    print_message(message)
                elif user_input=='3':
                    self.drop_cols(col)
                else:
                    message = 'Invalid option.'
                    print_message(message)
            except:
                print(col)
    def count_na(self, cols=None):
        """
        Counts number of nan's in the data set.
        =====================
        parameters

        cols = string or array. default=None.

        =====================
        output
        
        None
        """

        if cols == None:
            print(self.data.isna().sum().sort_values(ascending=False))
        else:

            if self.check_col_name(cols):

                print("Number of nulls: ", self.data[cols].isna().sum())


    def counts(self, col):
        """
        Prints the count numbers of values in a given pd.Series
        =====================
        parameters

        cols = string. Name of a column in the dataset.

        =====================
        output
        
        None
        
        """
        print(self.data[col].value_counts(dropna=False, normalize=True))

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

        fig, axes = plt.subplots(ncols=3, figsize=(18,6))
        
        data =self.data[col].copy()
        sns.distplot(data, bins='auto', ax=axes[0], label=f'Original (n={len(data)})')
        axes[0].set(title=f'Original (n={len(data)})')

        zscore_mask = self.zscore_outlier(data)
        data_zscore = data[~zscore_mask]
        sns.distplot(data_zscore, bins='auto', ax=axes[1], label=f'Zscore Outlier (n={len(data_zscore)}, {round((len(data_zscore)/len(data)*100),2)}%)')
        axes[1].set(title=f'Zscore Outlier (n={len(data_zscore)}, {round((len(data_zscore)/len(data)*100),2)}%)')

        iqr_mask = self.IQR_outlier(data)
        data_iqr = data[~iqr_mask]
        sns.distplot(data_iqr, bins='auto', ax=axes[2], label=f'IQR Outlier (n={len(data_iqr)}, {round((len(data_iqr)/len(data)*100),2)}%)')
        axes[2].set(title=f'IQR Outlier (n={len(data_iqr)}, {round((len(data_iqr)/len(data)*100),2)}%)')

        plt.suptitle(f'{col} distribution graphs', size=20)
        plt.show()
        option = input("Choose an option (1) none (2) zscore method (3) IQR method (4) Drop column: ")
        if option == '1':
            print('nothing has changed.')
        elif option == '2':
            self.update_data(self.data[~zscore_mask])
        elif option == '3':
            self.update_data(self.data[~iqr_mask])
        elif option == '4':
            self.drop_cols(col)
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
            - 'mode' will replace missing values by its mode
            - 0 will replace missing values by 0
            - str will replace missing values by given string
        ===================
        output:

        pd.Series
        """

        if self.check_col_name(col):
            if option == 'median':
                self.data[col].fillna(self.data[col].median(), inplace=True)
            elif option == 'mean':
                self.data[col].fillna(self.data[col].mean(), inplace=True)
            elif option == 'mode':
                self.data[col].fillna(stats.mode(self.data[col])[0][0], inplace=True)
            elif option == 0:
                self.data[col].fillna(0, inplace=True)
            else:
                self.data[col].fillna(option)
            print(self.count_na(col))

    def check_col_name(self, cols):
        """
        Checkes whether a given column or columns is(are) part of the dataframe.
        ==================
        input parameters:

        col : string. column name that needs to be immuted

        ===================
        output:

        boolean.
            - True if cols are included in the dataframe
            - False if cols are NOT included in the dataframe
        """
        if type(cols) == str:
            cols = [cols]
        if set(cols).issubset(set(self.data.columns)):
            return True
        else:
            message = f'{cols} not found in the data. Try again.'
            print_message(message)
            return False

    def set_cols(self, cat=[], cont=[]):
        """
        Manually define categorical and continuous columns bypassing self.col_classifier
        =====================
        parameters

        cat = list. List of categorical column names. Default=[]. 

        cont = list. List of continuous column names. Default=[]. 
        """
        if self.check_col_name(cat + cont):
            self.cat_cols = cat
            self.cont_cols = cont

            list_col = self.cat_cols + self.cont_cols
            drop_col = set(self.data.columns) - set(list_col)
            print(list(drop_col))
            self.drop_cols(list(drop_col))
        self.__str__()


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
        try:
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
        except:
            print('Error occured')
        
        plt.suptitle(f'Different scalers for {col} column (n={len(data)})', fontsize=20)
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

    def corr_map(self, data, annot=True, cmap='YlGnBu'):
        """
        Creates and prints a heatmap of correlation values between given data.
        ===================
        input parameters:
        
        data : pd.DataFrame

        annot : True by default. Can decide whether you would like to see the values on the heatmap graph.

        cmap : 'YlGnBu' by default. seaborn colormap themes
        =============
        output:

        correlation matrix
        """
        # if option == "cat":
        #     data = self.data[self.cat_cols].copy()
        #     title = 'Categorical Columns Correlation Heatmap'
        # elif option == "cont":
        #     data = self.data[self.cont_cols].copy()
        #     title = 'Continuous Columns Correlation Heatmap'
        # else:
        #     data = self.data[self.cont_cols + self.cat_cols].copy()
        #     title = 'All Columns Correlation Heatmap'
       
        # find correlation matrix between different variables given in data
        corr = data.corr()

        # creates triangular mask over the map
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        fig, ax = plt.subplots(figsize=(18,12))
        sns.heatmap(corr, annot=annot, cmap=cmap, mask=mask, ax=ax)
        ax.set(title='Correlation Heatmap')
        plt.show()
        return np.abs(corr * (~mask))

    def multicolinearity(self):
        """
        Allows the user to examine and get rid of possible multicolinearity variables using (1) correlation map and (2) 
        """
        options = """
                    Options:
                    \t1. Correlation Matrix
                    \t2. Variance Inflation Factor (VIF)
                  """
        print_message(options)
        method = input('Choose which method you would like to determine multicolinearity variables: ')
        self.choose_col_to_drop(method=method)
    
    def VIF(self):
        """
        Calculates and prints VIF
        """
        X = self.data[self.cont_cols + self.ohe_cols].drop(columns=self.target).copy()
        X = sm.add_constant(X)

        self.vif =pd.DataFrame()
        self.vif["variables"] = X.columns

        self.vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        print(self.vif.sort_values(by=['VIF'], ascending=False).head(10))
    
    def choose_col_to_drop(self, method):
        """
        Allows users to drop columns with a help of correlation matrix or VIF.
        ==================
        parameters 

        data : pd.DataFrame. Dataframe that you would like to check.

        method : string.
            - 1: correlation matrix / heatmap
            - 2: VIF
        ==================
        """
        # allows users to pick which columns to get rid of according to their correlation values
        
        # prints and creates correlation heatmap/matrix or VIF
        if method == '1':
            while True:
                cor = self.corr_map(self.data[self.cat_cols + self.cont_cols])
                print(cor.unstack().sort_values(ascending=False).head(10))
                print('\n')
                
                user_input = input('Write column name you would like to get rid of (Enter "x" to exit): ')
                if user_input in self.data.columns:
                    self.drop_cols(cols=user_input)
                elif user_input == 'x':
                    print_message(f"Exiting!\nThere are {self.data.shape[1]} columns remaining in the data")
                    break
                else: 
                    message = f"'{user_input}' is not found. Please try again."
                    print_message(message)
        
        elif method == '2':
            while True:
                self.VIF()
                print('\n')
                user_input = input('Write cut-off value (Usually 7, x-exit): ')
                if user_input == 'x':
                    print_message(f"Exiting!\nThere are {self.data.shape[1]} columns remaining in the data")
                    break
                else:
                    drop = list(self.vif[self.vif['VIF']>float(user_input)].variables)
                    # print(drop)
                    drop.remove('const')
                    # print(drop)
                    self.drop_cols(drop)

    def p_value_cut_off(self):
        p_value = pd.DataFrame(self.model.pvalues).reset_index()
        p_value.columns = ['variables', 'p_value']
        print(p_value.sort_values(by=['p_value'], ascending=False).head(10))
        while True:
                user_input = input('Write cut-off p-value (Usually 0.05, Enter "x" to exit): ')

                if user_input == 'x':
                    print_message(f"Exiting!\nThere are {self.data.shape[1]} columns remaining in the data")
                    break
                else:
                    drop = list(p_value[p_value['p_value'] > float(user_input)].variables)
                    # print(drop)
                    if 'const' in drop:
                        drop.remove('const')
                    # print(drop)
                    self.drop_cols(drop)
                    break
        

    
    def drop_cols(self, cols, ohe=False, verbose=True):
        """
        drops column(s)
        ==================
        parameters

        col = string or list containing the name(s) of the columns that needs to be deleted

        ohe = boolean. Default = False. True only if dropping the column was resulted from ohe.

        ohe_clear = boolean. Default = False. True if and only if used by self.drop_ohe_cols.

        verbose = boolean. Default = True. Prints out information about deleted columns.
        ==================
        output

        None
        """
        # check if col exists in the data column.
        if type(cols) == str:
            if self.check_col_name(cols):
                # self.data.drop(columns=[cols],inplace=True)
                if not ohe:
                    self.dropped_columns.append(cols)
                else:
                    self.ohe_dropped_columns.append(cols)
                if verbose:    
                    print_message([f"'{cols}' has been DELETED!", f"There are now {self.data.shape[1]} columns in the data."])

                # also updates cat/cont columns
                if cols in self.cat_cols:
                    self.cat_cols.remove(cols)
                elif cols in self.cont_cols:
                    self.cont_cols.remove(cols)

        elif type(cols) == list:
            if self.check_col_name(cols):
                self.data.drop(columns=cols, inplace=True)
                # also updates cat/cont columns
                for col in cols:
                    if col in self.ohe_cols:
                        self.ohe_cols.remove(col)
                    else: 
                        self.dropped_columns.append(col)
                        if col in self.cat_cols:
                            self.cat_cols.remove(col)
                            print('imhere')
                        elif col in self.cont_cols:
                            print('cont')
                            self.cont_cols.remove(col)
                    
                if verbose:
                    print_message([f"{len(cols)} columns have been DELETED!", f"There are now {self.data.shape[1]} columns in the data."])
        else:
            print_message("Invalid input!")

    def fix_col_names(self):
        """
        Fixes column names so that they can be used in OLS formula.
        """
        self.cat_cols = [name.replace('.','') for name in self.cat_cols]
        self.ohe_cols = [name.replace('.','') for name in self.ohe_cols]
        self.data.columns = [name.replace('.','') for name in self.data.columns]

    def ohe(self, cols=[], inplace=True):
        """
        One-Hot-Encoder using pandas built-in method.
        ======================
        parameters

        cols: array. list of columns names. Default = []
            - if left empty then goes through each cat_cols for the user to determine to use ohe to each column.
        inplace: boolean. Replaces the original if true.
        ======================
        output

        new pd.DataFrame with ohe columns
        """
        # Refreshes
        self.drop_ohe_cols(verbose=False)

        if len(cols)==0:
            message = "*** OHE TIME ***"
            print_message(message)

            # copies categorical column names
            cols = self.cat_cols.copy()
            for col in cols:
                self.get_col_info(col)
                while True:
                    user_input = input(f"Would you like to ohe column '{col}' (1-yes, 2-no, x-exit): ")
                    if user_input == '1':
                        df_dummies = pd.get_dummies(data=self.data[col], columns=[col], drop_first=True, prefix=col)
                        # self.drop_cols(col, ohe=True, verbose=False)
                        self.data = pd.concat([self.data, df_dummies], axis=1)
                        print_message([f"Column {col} has been one-hot-encoded", f"{df_dummies.shape[1]} columns have been added.",f"Total number of columns: {self.data.shape[1]}"])
                        self.ohe_cols += list(df_dummies.columns)
                        break
                    elif user_input == '2':
                        self.drop_cols(col)
                        break
                    elif user_input == 'x':
                        break
                    else:
                        message = 'Invalid input! Try again!'
                        print_message(message)
        else:
            if self.check_col_name(cols):
                return pd.get_dummies(data=self.data, columns=cols, drop_first=True, prefix=cols)
    def test(self):
        print('testing')
    
    def drop_ohe_cols(self, verbose=True):
        # Removes all ohe columns
        self.drop_cols(self.ohe_cols, ohe=True, verbose=verbose)
        # return dropped ohe columns back to categorical columns
        self.cat_cols += self.ohe_dropped_columns
        self.ohe_cols = []
        # Set ohe dropped columns to an empty list
        self.ohe_dropped_columns = []

    def get_col_info(self, col):
        """
        gets and print out column information
        =======================
        parameter

        col = string. column name.
        =======================
        output

        None
        """
        if self.check_col_name(col):
            messages = [
                f"Column Name: {col}",
                f"Number of unique values: {self.data[col].nunique()}",
                self.data[col].value_counts(normalize=True)
                        ]
            print_message(messages)

    def split(self, option=None, train_size=0.75, shuffle=True, random_state=42):
        """
        splits the data into test and train data set using sklearn.model_selection
        ===================
        parameters

        option = string.
            - None = both categorical and continuous columns are used
            - 'cat' = only categorical columns are used
            - 'cont' = only continuous columns are used

        train_size : float. Determines how much percent of original data you would want as train set. (Default = 0.75)

        shuffle : boolean. (Default = True)

        random_state : int. random state (Default = 42)
        ===================
        output

        X_train, X_test, y_train, y_test
        """
        # Sets X and y
        if option == 'cat':
            self.X = self.data[self.ohe_cols]#.drop(columns=[self.target]) 
        elif option == 'cont':
            self.X = self.data[self.cont_cols].drop(columns=[self.target]) 
        else:
            self.X = self.data[self.cont_cols + self.ohe_cols].drop(columns=[self.target]) 
        self.y = self.data[self.target]

        # Splits data into train and test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=train_size, shuffle=shuffle, random_state=random_state)

        # prints out resulting shapes
        messages = [f'Shape of X_train: {self.X_train.shape}', f'Shape of X_test: {self.X_test.shape}', f'Shape of y_train: {self.y_train.shape}', f'Shape of y_test: {self.y_test.shape}']
        print_message(messages)
        self.splitted = True

    def get_formula(self):
        """
        creates formula that can be used in OLS regression.
        =====================
        output

        formula : string.
        """

        features = " + ".join(self.X_train.columns)
        
        formula = self.target + ' ~ ' + features

        return formula

    def regression(self):
        """
        Creates a regression model using OLS.
        """
        if self.splitted:
            # Gets a formula that can be used in OLS
            formula = self.get_formula()
            print_message(["Formula:", formula])

            # Train

            # initialize self.model
            data = pd.concat([self.X_train, self.y_train], axis=1)
            self.model = smf.ols(formula=formula, data=data).fit()

            return self.model
        else:
            message = "Data has NOT been splitted yet. Please do it first!"
            print_message(message)


    def validate_model(self):
        """
        Validates the model via
        1. QQ plots - check for normality of residuals
        2. Residual graphs - check for homoscedesticity
        3. Compare MSE and r2_score for both train and test sets for overfitting/underfitting
        """
        self.y_hat_train = self.model.predict(self.X_train) 
        self.y_hat_test = self.model.predict(self.X_test)

        self.resid_train = self.y_hat_train - self.y_train 
        self.resid_test = self.y_hat_test - self.y_test
        
        self.train_mse = mean_squared_error(self.y_train, self.y_hat_train)
        self.test_mse = mean_squared_error(self.y_test, self.y_hat_test)

        self.train_r2 = r2_score(self.y_train, self.y_hat_train)
        self.test_r2 = r2_score(self.y_test, self.y_hat_test)

        ## QQ plots
        fig, axs = plt.subplots(ncols=2, figsize=(18,12))
        sm.graphics.qqplot(self.resid_train, line='45', fit=True, ax=axs[0])
        axs[0].set(title='QQ plot for training data set')
        sm.graphics.qqplot(self.resid_test, line='45', fit=True, ax=axs[1])
        axs[1].set(title='QQ plot for test data set')
        
        # Homoscedasticity
        fig, axs = plt.subplots(ncols=2, figsize=(18,12))
        sns.scatterplot(self.y_train, self.resid_train, ax=axs[0], label='Training Residuals')
        axs[0].set(title='Training Residual Graph', xlabel='y_train', ylabel='Residuals')
        axs[0].axhline(0, label='zero')

        sns.scatterplot(self.y_test, self.resid_test, ax=axs[1], label='Testing Residuals')
        axs[1].set(title='Test Residual Graph', xlabel='y_test', ylabel='Residuals')
        axs[1].axhline(0, label='zero')

        message=[f'Train MSE = {self.train_mse}\tTrain R2 = {self.train_r2}', f'Test MSE = {self.test_mse}\tTest R2 = {self.test_r2}']
        print_message(message)


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

def save_data(data, name):
    with open(name, 'wb') as f:
        pickle.dump(data, f)

def load_data(name):
    with open(name, 'rb') as f:
        return pickle.load(f)