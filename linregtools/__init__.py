"""Functions to iterate on common tasks related to exploring data to prepare it
for linear regression modeling, and performing the modeling itself.

By Jessica Miles.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from statsmodels.formula.api import ols
import statsmodels.api as sm

def assign_cats(df_groups, df_records, ratio_map={}, seed=5):
    """Assigns group labels and multipliers to records to create an
    experimental data set.
    
    Returns: the original mergeto dataframe with the group labels and 
    multipliers assigned in new columns.
    
    _____________________________
    Args:
    -----------------------------
    
    df_groups: a dataframe consisting of a group of categorical labels
        and corresponding multiplier column representing the weight of effect for
        that group. Group label should be in column 1 (index 0), and multiplier in 
        column 2 (index 1).
        
    df_records: a dataframe representing records to which the groups will be
        assigned. Records df should have more rows than groups, where groups will
        be assigned to batches of records.
        
    ratio_map: default, none. A dict consisting of keys = labels and 
        values = proportion of records to be assigned that label. Proportions
        should sum to 1. 
        
        - If a ratio map is passed, this will be used to determine
            group sizes instead of making them roughly equal and assigning to labels
            randomly.
        
        - If no ratio map passed, determines roughly even but varied group sizes 
        based on number of groups and number of total records, and assigns the 
        group labels and multipliers to the records randomly.
        
    seed: default 5. Integer representing random seed for reproducability.
    """
    np.random.seed(seed)
    
    record_ct = len(df_records)
    group_ct = len(df_groups)
    groups_sizes = []
    group_index = []
    
    # apply group sizes using ratio map, if passed
    if len(ratio_map.items()) > 0:
        
        for i, v in enumerate(ratio_map.values()):
            if i == (group_ct - 1):
                remaining = record_ct - np.sum(groups_sizes)
                groups_sizes.append(np.around(remaining, decimals=0))
            else:
                groups_sizes.append(np.around((record_ct * v), decimals=0))
        
    else:
        # determine the group size if we were to divide evenly
        split = np.around((record_ct / group_ct), decimals=0)

        # loop for each group and assign it a randomly altered size
        while len(groups_sizes) < group_ct:

            # if last 4 groups, spread evenly between them
            if len(groups_sizes) == (group_ct - 4):
                remaining = record_ct - np.sum(groups_sizes)
                last4 = np.around((remaining / 4), decimals=0)
                for i in range(0, 3):
                    groups_sizes.append(last4)
                groups_sizes.append(record_ct - np.sum(groups_sizes))
            else:
                # generate a random int between size and 0 to alter the size
                alterby = np.random.randint(low=0, high=split, dtype=int)

                # add alter variable to half the size and append to list
                groups_sizes.append((np.around((split / 2), decimals=0)) + alterby)
    
    # loop through groups and create appropriate number of copies for
    # each group size
    for i, size in enumerate(groups_sizes):
        for j in range(0, int(size)):
            group_index.append(list(df_groups.iloc[i].values))
            
    # shuffle group index to re-randomize it
    np.random.shuffle(group_index)
    
    group_records = pd.DataFrame(group_index, columns=df_groups.columns)
    
    # concatenate dfs
    new_df = pd.concat([df_records, group_records], axis=1)

    return new_df

def explore_data(to_explore, df, target, hist=True, box=True, plot_v_target=True,
                 summarize=True, norm_check=True):
    """Creates plots and summary information intended to be useful in preparing
    for linear regerssion modeling. Prints plots of distributions, a scatter
    plot of each predictor column against a target, and outputs a dataframe of
    metadata including results of a normality check and correlation coefficient.

    Returns: a dataframe representing the metadata collected.
    _____________________________
    Args:
    -----------------------------
    
    to_explore: list of column names to explore
    
    df: Dataframe containing the columns in to_explore, as well as the target
        column
    
    target: string of the column name to use as the target, or dependent variable
    
    hist: True or False (default True). Whether to include a histogram for each
    predictor column.
    
    box: True or False (default True). Whether to include a box plot for each
    predictor column.
    
    plot_v_target: True or False (default True). Whether to include a scatter 
    plot showing the predictor versus target
    
    summarize: True or False (default True). Whether to include a summary of
    the values in each predictor column. Data will be summarized using 
    df.describe() for variables deemed continuous, and df.sort_values()
    for variables deemed categorical. Classification of continuous versus
    categorical is best effort.
    
    norm_check: True or False (default True). Whether to perform a normality
    check using SciPy's stats omnibus normality test. Null hypothesis 
    is that the data comes from a normal distribution, so a value less than
    0.05 represents likely NOT normal data.
    """
    
    # Create some variables to dynamically handle including/excluding 
    # certain charts
    num_charts = 0
    if hist:
        num_charts += 1
    if box:
        num_charts += 1 
    if plot_v_target:
        num_charts += 1
        
    # check if input column is a list; if not, make it one. This allows for
    # a string to be passed if only one column is being summarized.
    if type(to_explore) == str:
        temp_list = [to_explore]
        to_explore = temp_list
    
    # column headers for metadata output df
    meta_list = [['col_name', 'corr_target', 'assumed_var_type', 'omnibus_k2',
                 'omnibus_pstat', 'is_normal', 'uniques', 'mean', 'median']]
    
    # loop through each column in the list to analyze
    for col in to_explore:
        
        header_line = '-'*75
        header_text = f'\nExploring column: {col}\n'
        print(header_line + header_text + header_line)
        
        # Determine if categorical or continuous
        # assume continuous to begin with
        var_type = 'continuous'
        data_type = df[col].dtype
        uniques = np.nan
        mean = np.nan
        median = np.nan
        num_uniques = len(df[col].unique())
        
        if df[col].dtype in ['int64', 'float64']:
            # number types need the most analysis because they could be
            # categorical even if they're numeric
            
            # using 100 as an arbitrary cutoff here, may need adjustment
            if num_uniques < 20:
                var_type = 'categorical'
                uniques = num_uniques
            else:
                mean = np.mean(df[col])
                median = np.median(df[col])
        elif df[col].dtype in ['object']:
            # Assuming column types have been fixed at this point,
            # so if a column is not numerical it must be categorical
            var_type = 'categorical'
            uniques = num_uniques
        elif df[col].dtype in ['datetime64']:
            var_type = 'date'
            
        # print summary based on data type
        if summarize:
            if var_type in ['continuous', 'date']:
                header_text = f'\ndf.describe() for continuous data: {col}\n'
                print(header_line + header_text + header_line)
                print(df[col].describe())
            else:
                header_text = f'\nValue Counts for categorical data: {col}\n'
                print(header_line + header_text + header_line)
                with pd.option_context('display.max_rows', 20):
                    print(df[col].value_counts())
        
        # creates scatter plots, histogram, and box plots for numerical data
        if data_type in ['int64', 'float64']:
            if num_charts > 0:

                fig, axes = plt.subplots(nrows=num_charts, ncols=1, 
                                         figsize=(8, num_charts * 5))
                if hist:
                    if num_charts > 1:
                        ax1 = axes[0]
                    else:
                        ax1 = axes

                    if box:
                        ax2 = axes[1]
                        if plot_v_target:
                            ax3 = axes[2]
                    elif plot_v_target:
                        ax3 = axes[1]
                elif box:
                    if num_charts > 1:
                        ax2 = axes[0]
                    else:
                        ax2 = axes

                    if plot_v_target:
                        ax3 = axes[1]

                elif plot_v_target:
                    ax3 = axes


                # add a little extra space for headers
                plt.subplots_adjust(hspace=0.3)

                # Histogram
                if hist:
                    sns.histplot(df[col], kde=True, ax=ax1)
                    ax1.set_title(f"Hist {col}")

                # Box plot
                if box:
                    sns.boxplot(x=df[col], ax=ax2)
                    ax2.set_title(f"Boxplot {col}")

                # Plot against target
                # create a series representing quartiles, to use as hue
                if plot_v_target:                
                    if var_type == 'continuous':
                        try:
                            quartile_labels=['q1', 'q2', 'q3', 'q4']
                            quartiles = pd.qcut(df[col], 4, 
                                                labels=quartile_labels, 
                                                duplicates='drop')
                            sns.scatterplot(x=df[col], y=df[target], ax=ax3, 
                                            hue=quartiles)
                            ax3.legend(title=f'{col} quartiles')
                            
                        except:
                            sns.scatterplot(x=df[col], y=df[target], ax=ax3)
                    else:
                        sns.scatterplot(x=df[col], y=df[target], ax=ax3)
                    ax3.set_title(f"{col} versus {target}")
                    
                plt.show();
                
            # get pearson correlation coefficient between col and target
            corr = df[[col, target]].corr()

            # Test for normality using scipy omnibus normality test
            # null hypothesis is that the data comes from a normal distribution
            if norm_check:
                k2, p = stats.normaltest(df[col])
                if p < 0.05:
                    normal = False
                    print(f'\nData is NOT normal with p-statistic = {p}\n')
                else:
                    normal = True
                    print(f'\nData IS normal with p-statistic = {p}\n')

            # append metadata to list of lists
            meta_list.append([col, corr.iloc[0][1], var_type, k2, p, normal, 
                uniques, mean, median])
            
        # Create catplot for categorical data
        elif data_type in ['object', 'str']:
            # get variable to determine appropriate height based on number
            # of categories to be displayed
            h = len(df[col].value_counts())
            
            # Get list of categories sorted in alpha order
            order = df[col].unique()
            order.sort()
            
            fig, ax = plt.subplots(figsize=(8, (h*0.15)+4))
            sns.barplot(x=target, y=col, data=df, orient='h', 
                        order=order, ax=ax)
            ax.set_title(f"Average {target} per {col}");
            plt.show();
        
    df_meta = pd.DataFrame(data=meta_list[1:], columns=meta_list[0])
    return df_meta

def preprocess(df, target, cont_cols, cat_cols=None, cat_drop=None, 
               standardize=True, std_cat=False, ttsplit=True, test_prop=0.25,
               random_state=5, verbose=False):
    """Takes a dataframe and preprocesses it for linear regression modeling.
    
    Returns X_train, X_test, y_train, y_test dataframes after transformation.
    If ttsplit is False, returns X_all, y_all dataframes after transformation.
    
    First, if instructed, does a train-test-split.
    
    If provide cat_cols, also performs one-hot-encoding using OneHotEncoder
    on categorical variables. Drops one of each category columns as directed.
    
    Finally, standardizes the independent variables using StandardScaler. 
    If split into train and test, StandardScaler fit_transforms on train and
    uses the same fit parameters to transform test.
    
    This does NOT fill null values or drop outliers. If you wish to do this, 
    it should be done on data that has already been split for training and
    testing.
    
    _____________________________
    Args:
    -----------------------------

    df: a dataframe including all rows and columns. If it contains columns
    not specified in target, contcols, or catcols, these will be ignored
    in terms of processing steps.
    
    target: string representing column name of the target variable. This will 
    not be transformed, but will be split into training and testing.
    
    cont_cols: list of continuous columns. These will not be OHE.
    
    cat_cols: list of categorical columns. These are the columns that will be
    OHE. These may be string/text values, or may even be numerical.
    Note that if you want to apply OHE, you must provide at least this list
    of cat cols. If you do not provide the cat_cols, no OHE will be performed.
    
    cat_drop: Default is None. Can also be a list of categories to be dropped, 
    where categories are named like column_category, the way OneHotEncoder will
    atomatically name the columns.
    You may also specify 'first' instead of providing a list to have OHE
    automatically drop the first category for you, or you may specify None
    to have no categories be dropped.
    
    standardize: Default is True. If you do not wish to standardize data, 
    specify False. 
    Only X columns will be standardized; y (target) will be returned as-is.
    If performed OHE, the OHE columns will also be standardized.

    std_cat: Default is False. Indicates whether you want to standardize OHE
    variables along with continuous. Only interpreted if standardize is True.
    
    ttsplit: Default is True. If you do not wish to perform a train test split,
    specify False. In this case all rows of transformed x columns will be
    returned along with y.
    
    test_prop: Default is 0.25. Represents the proportion of rows that should
    be designated for the test data set versus training. Training will be
    assigned 1-test_prop.
    
    random_state: Default is 5. Random seed to use for reproducability.
    -----------------------------------
    """
    
    X_train, X_test, y_train, y_test = None, None, None, None
    
    # Combine continuous and categorical columns into a single x_cols
    x_cols = cont_cols.copy()
    if cat_cols is not None:
        x_cols.extend(cat_cols)
    
    # Perform train-test-split, if indicated
    if ttsplit:
        if verbose:
            print(f'Performing train-test split...')
        # perform split
        X_train, X_test, y_train, y_test = train_test_split_custom(
                            df=df, 
                            target=target, 
                            x_cols2=x_cols,
                            test_prop=test_prop,
                            random_state=random_state,
                            verbose=verbose)
    else:
        # if didn't specify to do train-test-split, compile full dfs based
        # on specified columns
        X_train = df[x_cols].copy()
        y_train = df[target].copy()
    
    # perform One Hot encoding
    if cat_cols is not None:
        if verbose:
            print(f'Performing OHE...')
            print(f'cat_cols = {cat_cols}')
            print(f'cat_drop = {cat_drop}')
        
        def process_ohe(df, cat_cols, cat_drop):
            # get OHE version
            df_ohe = ohe_custom(df[cat_cols], cat_drop=cat_drop)
            # concat non-cat columns from original df with ohe columns
            df.reset_index(drop=True, inplace=True)
            df.drop(columns=cat_cols, inplace=True)
            df_final = pd.concat([df, df_ohe], axis=1, copy=False)
            return df_final
        
        X_train = process_ohe(X_train, cat_cols, cat_drop)
        
        if X_test is not None:
            X_test = process_ohe(X_test, cat_cols, cat_drop)
        
        if verbose:
            print(f'Columns after OHE: {X_train.columns}')
            
    # perform standardization, if indicated
    if standardize:
        if verbose:
            print(f'Performing standardization...')

        scaler = StandardScaler()

        if std_cat:
            # if we're standardizing all x columns
            X_train = pd.DataFrame(scaler.fit_transform(X_train), 
                                   columns=X_train.columns)
            if X_test is not None:
                # we're transforming test using the same fit parameters from train
                X_test = pd.DataFrame(scaler.transform(X_test), 
                            columns=X_test.columns)
        else:
            # if we're standardizing only continuous columns
            X_train_cont = pd.DataFrame(scaler.fit_transform(X_train[cont_cols]), 
                columns=X_train[cont_cols].columns)
            X_train = pd.concat([X_train_cont, X_train.drop(columns=cont_cols)], 
                axis=1, copy=False)

            if X_test is not None:
                # we're transforming test using the same fit parameters from train
                X_test_cont = pd.DataFrame(scaler.transform(X_test[cont_cols]), 
                    columns=X_test[cont_cols].columns)
                X_test = pd.concat([X_test_cont, X_test.drop(columns=cont_cols)], 
                    axis=1, copy=False)
            if verbose:
                print("X_train after standardization:")
                print(X_train.describe())
                print("X_test after standardization:")
                print(X_test.describe())
        
    # return final dfs
    if X_test is not None:
        return X_train, X_test, y_train, y_test
    else:
        return X_train, y_train

def train_test_split_custom(df, target, x_cols2, 
                            test_prop=0.25, random_state=5, verbose=False):
    
    if verbose:
        print(f'Performing ttsplit with random state {random_state}')
    X_train, X_test, y_train, y_test = train_test_split(df[x_cols2], 
                                df[target], test_size=test_prop,
                                random_state=random_state)
    
    return X_train, X_test, y_train, y_test

def ohe_custom(df, cat_drop=None):
    
    if cat_drop == 'first' or cat_drop == None:
        drop = cat_drop
    else:
        drop = None
        
    ohe = OneHotEncoder(drop=drop)
    ohe_arr = ohe.fit_transform(df).todense()

    cols = ohe.get_feature_names(df.columns)
    df_ohe = pd.DataFrame(ohe_arr, columns=cols)
    
    if type(cat_drop) == list:
        df_ohe.drop(columns=cat_drop, inplace=True)
    
    return df_ohe

def rss(y_actuals, y_preds):
    """Calculates the Residual Sum of Squares (RSS) a.k.a. Sum of Squared
    Residuals (SSR) a.k.a Squared Sum of Errors (SSE) between predicted and
    actual target values.
    """
    return np.sum((y_actuals - y_preds )**2)

def iterate_models_sklearn(model_dict, df, target, cont_cols2, all_cat_cols, 
                           avg_cats, std_cat=False, random_state=5, verbose=False):
    """Uses a dictionary of different model parameters to process and fit a 
    a series of different linear regression models on the same base dataset
    using scikit-learn. Currently tailored to this specific dataset 
    (with 'dotw', for instance) but could be easily modified to be more general.
    
    Returns dataframes of resulting coefficients from each model, as well as
    accuracy statistics for each model.
    
    _____________________________
    Args:
    -----------------------------

    model_dict: dictionary of dictionaries representing different params to
    preprocess the data before feeding to the model.
    
    df: dataframe of all data
    
    target: string of target column name
    
    cont_cols: list of continuous columns, to be standardized if indicated.
    
    all_cat_cols: list of all categorical columns. Assumes dotw is the last 
    element, and it will be sliced off if not to be included.
    
    avg_cats: list of the categories that represent the averages. Should be in
    the format of colname_category, which is how OneHotEncoder will name them.

    std_cat: Default is False. Indicates whether you want to standardize OHE
    variables along with continuous. Only interpreted if standardize is True.
    
    random_seed: random seed to use for reproducability.
    """
    
    results_coefs = []
    results_stats = []


    # loop through models with different parameters, and get results  
    for mod_id, model in model_dict.items():
        
        cat_drop = []
        cat_cols = []

        # determine category behavior based on model parameters
        if model['keep_dotw']:
            cat_drop = avg_cats if model['cat_drop'] == 'avg' else model['cat_drop']
            cat_cols = all_cat_cols
        else:
            cat_drop = avg_cats[:-1] if model['cat_drop'] == 'avg' else model['cat_drop']
            cat_cols = all_cat_cols[:-1]
            
        if verbose:
            header_line = '-'*75
            header_text = f'\nInitiating processing for model: {mod_id}\n'
            text = f"""\nModel params: {model}
            \nCategorical columns to include: {cat_cols}
            \nCategories to drop from model: {cat_drop}
            \nContinuous columns: {cont_cols2}\n"""
            print(header_line + header_text + header_line + text + header_line)
            print(f'Preprocessing dataset with random state {random_state}...')

        # preprocess according to model parameters
        X_train, X_test, y_train, y_test = preprocess(
            df, target, cont_cols2, cat_cols=cat_cols, cat_drop=cat_drop, 
            standardize=model['stand'], std_cat=std_cat, random_state=random_state, 
            verbose=verbose)
        
        if verbose:
            print(f'Preprocessing complete.\nFitting model...')

        # fit linear regression to preprocessed training data
        linreg = LinearRegression(fit_intercept=True)
        linreg.fit(X_train, y_train)
        y_pred_train = linreg.predict(X_train)
        y_pred_test = linreg.predict(X_test)
        
        if verbose:
            print(f'Model at mem address {id(linreg)} fit.\n')
            print(f'Y-intercept: {linreg.intercept_}\n')
            print(f'Coefficients: {linreg.coef_}\n')

        # set up to prepare to record results
        coef_dict = {'model_id': mod_id}
        acc_dict = {'model_id': mod_id}
        x_cols3 = X_train.columns

        # record coefficients and y-intercept
        coef_dict['y-intercept'] = linreg.intercept_
        for i, coef in enumerate(linreg.coef_):
            coef_dict[x_cols3[i]] = coef

        # append coefficient results for this model to overall list
        results_coefs.append(coef_dict)
        
        if verbose:
            print(f'Gathering accuracy stats...\n')

        # get statistics for model performance
        acc_dict['R-sq Train'] = linreg.score(X_train, y_train)
        acc_dict['R-sq Test'] = linreg.score(X_test, y_test)
        acc_dict['RMSE Train'] = metrics.mean_squared_error(y_true=y_train, 
                                y_pred=y_pred_train, squared=False)
        acc_dict['RMSE Test'] = metrics.mean_squared_error(y_true=y_test, 
                                y_pred=y_pred_test, squared=False)
        acc_dict['RSS Train'] = rss(y_train, y_pred_train)
        acc_dict['RSS Test'] = rss(y_test, y_pred_test)

        # append statistic results for this model to overall list
        results_stats.append(acc_dict)
        
        if verbose:
            print(f'Model {mod_id} Complete!\n')
            print(f'Continuous columns: {cont_cols2}\n')

    # turn results lists into dataframes
    df_coefs = pd.DataFrame(results_coefs)
    df_stats = pd.DataFrame(results_stats)
    
    return df_coefs, df_stats

def currency(x, pos=None):
    """Formats numbers as currency, including adding a dollar sign and abbreviating numbers
    over 1,000. Can be used to format matplotlib tick labels.
    _____________________________
    Args:
    -----------------------------
        x (integer or float): Number to be formatted.
        pos (optional): Included for matplotlib, which will use it. Defaults to None.

    Returns:
        string: formatted string based on number.
    """
    # over 1 billion
    if abs(x) >= 1000000000:
        return '${:1.2f} B'.format(x*1e-9)
    # over 10 million
    elif abs(x) >= 10000000:
        return '${:1.1f} M'.format(x*1e-6)
    # over 1 million
    elif abs(x) >= 1000000:
        return '${:1.2f} M'.format(x*1e-6)
    elif x == 0:
        return '${:0}'.format(x)
    elif abs(x) >= 1000:
        return '${:1.1f} K'.format(x*1e-3)
    else:
        return '${:.1f}'.format(x)

def iterate_model_sm(target, x_cols, df):
    """Uses statsmodels to fit an OLS linear regression model to data provided.
    Prints the summary of the model, displays a qq-plot to check the normality
    of residuals, and plots predictions against residuals to check for
    homoscedasticity.
    _____________________________
    Args:
    -----------------------------
    target: string, name of the column in the dataframe that represents the
    dependent, or target variable.
    
    x_cols: list of column names to be included as independent, or predictor
    variables in the model.
    
    df: Dataframe including x_cols and target column. Any additional columns
    included in df which are not included in x_cols or target will be ignored.
    Data should be standardized and dummied as appropriate before calling this
    function, since it does not perform these tasks.
    """
    
    x_count = 1
    
    # Create formula
    if type(x_cols) == str:
        formula = f'{target}~' + x_cols
    else:
        if len(x_cols) > 1:
            formula = f'{target}~' + '+'.join(x_cols) 
            x_count = len(x_cols)
        else:
            formula = f'{target}~' + x_cols[0]

    # Fit model using statsmodels
    model = ols(formula=formula, data=df).fit()
    print(model.summary())

    # Create a qq plot to check for normality of the residuals
    fig, ax = plt.subplots()
    fig = sm.qqplot(model.resid, dist=stats.norm, line='45', ax=ax, fit=True)
    ax.set_title("QQ Plot for Normality Check")
    plt.show();

    # Plot residuals to check for homoscedasticity
    fig, ax = plt.subplots(figsize=(8, 6))
    if x_count > 1:
        #dummy_x = np.arange(1, len(df)+ 1)
        dummy_x = model.predict(df[x_cols])
        sns.scatterplot(x=dummy_x, y=model.resid, ax=ax)
    else:
        sns.scatterplot(x=df[x_cols], y=model.resid, ax=ax)
    ax.set_title("Residuals Plot for Homoscedasticity")
    plt.axhline(y=0, color='black')
    plt.show();
    return model