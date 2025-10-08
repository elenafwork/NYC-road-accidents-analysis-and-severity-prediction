# libraries
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import contextily as ctx
import numpy as np
import holidays
import plotly.graph_objs as go
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import plotly.express as px
from typing import List, Tuple, Optional
import seaborn as sns
from IPython.display import display


# function to load dataset
def load_csv(file_path):
    """
    Load CSV file with specified dtype and memory config.
    :file_path: extended file path
    """
    extended_path = os.path.expanduser(file_path)
    df = pd.read_csv(extended_path, dtype={'ZIP CODE': str}, low_memory=False)
    print(f"Loaded: {file_path}")
    return df


# dataset 
def inspect_data(df):
    """
    Print column types, missing values, and sample categorical stats.
    """
    print("Column Types:")
    print(df.dtypes)
    print("\nMissing Values (Top 10):")
    print(df.isnull().sum().sort_values(ascending=False).head(10))
    print("\nTop 5 Boroughs:")
    print(df['BOROUGH'].value_counts().head())
    print("\nTop 5 Contributing Factors:")
    print(df['CONTRIBUTING FACTOR VEHICLE 1'].value_counts().head())

# preprocess for EDA
def preprocess_for_eda(df):
    """
    Add key engineered columns used in EDA.
    """
    df['CRASH DATE'] = pd.to_datetime(df['CRASH DATE'], errors='coerce')
    df['DAY_OF_WEEK'] = df['CRASH DATE'].dt.day_name()
    df['SEVERITY_SCORE'] = df[['NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED']].sum(axis=1)
    df['BOROUGH'] = df['BOROUGH'].fillna("Unknown")
    return df
# ------------------------------------------------------


# checking the missing values and what percentage of the set
def missing_values(df):
    """
    callculates the sum and rate of missing values(%)
    for each column and print them
    """
    missing_values=df.isna().sum()
    print(missing_values)
    print("\nMissing value percentages per column:")
    missing_percentages =missing_values / len(df) * 100
    print(missing_percentages)
    
# dropping columns
def drop_columns(df, cols_to_drop):
    """
    Drop specified columns from the DataFrame.

    :param df: pandas DataFrame
    :param cols_to_drop: list of column names to drop
    :return: DataFrame with specified columns removed
    """
    df = df.copy()
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    print(f"Dropped columns: {', '.join(cols_to_drop)}")
    print(f'Dataset shape: {df.shape[0]}  rows and {df.shape[1]}  columns')
    return df

# adding feature CRASH_TYPE
def add_crash_type(df):
    """
    Add a CRASH_TYPE column based on number of persons killed or injured.
    
    Rules:
      - 'Fatal'     if NUMBER OF PERSONS KILLED > 0
      - 'Injury'    if NUMBER OF PERSONS INJURED > 0
      - 'No victims' otherwise
    
    Prints:
      - Counts of each crash type
      - Percentages of each crash type
    """
    df = df.copy()

    df['CRASH_TYPE'] = df.apply(
        lambda row: 'Fatal' if row['NUMBER OF PERSONS KILLED'] > 0
        else ('Injury' if row['NUMBER OF PERSONS INJURED'] > 0 else 'No victims'),
        axis=1
    )

    # Counts
    crash_type_count = df['CRASH_TYPE'].value_counts()
    print("Crash type counts:\n", crash_type_count)

    # Percentages
    crash_type_pr = (crash_type_count / crash_type_count.sum()) * 100
    print("\nCrash type percentages:\n", crash_type_pr.round(2))

    return df

# feature engineering 
def feature_engineering(df):
    """
    Adds:
      - CRASH DATETIME, HOUR, DAY_OF_WEEK, DAY_NAME, YEAR, MONTH
      - IS_WEEKEND
      - IS_PUBL_HOLIDAY
      - IS_HIGHWAY
      - IS_BRIDGE (word-boundary match for 'bridge')
      - NUM_DRIVERS
    """
    df=df.copy()
    # -----DATE AND TIME FEATURES
    df['CRASH DATETIME'] = pd.to_datetime(df['CRASH DATE'] + ' ' + df['CRASH TIME'], errors='coerce')
    df['HOUR'] = df['CRASH DATETIME'].dt.hour
    df['DAY_OF_WEEK'] = df['CRASH DATETIME'].dt.dayofweek
    df['DAY_NAME'] = df['CRASH DATETIME'].dt.day_name()
    df['YEAR'] = df['CRASH DATETIME'].dt.year
    df['MONTH'] = df['CRASH DATETIME'].dt.month
    print('Date and Time features are successfully extracted!')
    print('Columns CRASH DATETIME, HOUR, DAY_OF_WEEK, DAY_NAME, YEAR, MONTH are succesfully added!')

    # --------weekend flag
    df['IS_WEEKEND']=df['DAY_OF_WEEK'].apply(lambda x:1 if x>=5 else 0)
    print('Feature IS_WEEKEND is succesfully added!')
    
    # ----Get U.S. federal holidays (you can also specify state, e.g., state='NY')
    us_holidays = holidays.US(years=df['CRASH DATETIME'].dt.year.unique())
    # ------ flag for public holidays
    df['IS_PUBL_HOLIDAY'] = df['CRASH DATETIME'].apply(lambda x: 1 if x in us_holidays else 0)
    print('feature IS_PUBL_HOLIDAY is successfully added!')
    
    # ----HIGHWAY flag, CODED 1/0
    name = df['ON STREET NAME'].fillna('').str.lower()
    search_keywords = [ 'parkway','tollway', 'hwy', 'highway', 'freeway', 'expwy', 'expwy.', 'expressway', 'route', 'rt', 'interstate', 'i-', 'us-',                           'sr-', 'ca-', 'ga-', 'tx-']
    # ------Safely apply logic, skipping NaNs
    df['IS_HIGHWAY'] = df['ON STREET NAME'].str.lower().fillna('').apply(    lambda x: 1 if any(keyword in x for keyword in search_keywords) else 0 )
    print('feature IS_HIGHWAY is successfully added!')

    # ---bridge flag
    bridge_pattern = r'\bbridge\b'
    df['IS_BRIDGE'] = name.str.contains(bridge_pattern, regex=True, na=False).astype(int)
    print('Feature IS_BRIDGE added')

    #------number of drivers
    vehicle_cols = [col for col in df.columns if 'VEHICLE TYPE CODE' in col.upper()]

    # Count how many of those are non-null per row (i.e., number of drivers)
    df['NUM_DRIVERS'] = df[vehicle_cols].notna().sum(axis=1)
    print('feature NUM_DRIVERS is successfully added!')

    return df
    
# FILL BOROUGN with HIGHWAY/BRIDGE
def fill_borough(df):
    """
    Fill missing BOROUGH values using logic based on IS_BRIDGE and IS_HIGHWAY
    """
    
    df['BOROUGH'] = df.apply(
        lambda row: 'BRIDGE' if pd.isna(row['BOROUGH']) and row['IS_BRIDGE'] == 1
        else ('HIGHWAY' if pd.isna(row['BOROUGH']) and row['IS_HIGHWAY'] == 1
        else ('UNKNOWN' if pd.isna(row['BOROUGH']) else row['BOROUGH'])),
        axis=1
    )
    print('column BOROUGH is filled for missing values!')
    return df

def add_severity_class(df, type_col='CRASH_TYPE', new_col='SEVERITY_CLASS'):
    """
    Map crash type categories to numeric severity levels.

    Default mapping:
        'No victims' -> 1
        'Injury'     -> 2
        'Fatal'      -> 3

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the crash type column.
    type_col : str
        Name of the crash type column in df.
    new_col : str
        Name of the new numeric column to add.

    Returns
    -------
    pd.DataFrame
        DataFrame with the new severity class column added.
    """
    severity_map = {
        'No victims': 0,  
        'Injury': 1,      
        'Fatal': 1       
    }
    
    df = df.copy()
    df[new_col] = df[type_col].map(severity_map)
    
    print(f"Binary column '{new_col}' added based on '{type_col}':")
    print(df[new_col].value_counts(normalize=True))
    
    return df






### PLOTS----------------------------------------------------
def visualize_crash_distribution(df):
    """
    Create visualizations of crash fatalities by year and crash type.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'YEAR' and 'CRASH_TYPE' columns
    
    Returns:
    --------
    tuple : A tuple containing two matplotlib figure objects
        - First figure: Absolute numbers bar chart
        - Second figure: Percentage distribution horizontal bar chart
    
    Notes:
    ------
    - Requires pandas and matplotlib to be imported
    - Assumes 'YEAR' and 'CRASH_TYPE' are valid columns in the DataFrame
    """
       
    # ------Group by YEAR and CRASH_TYPE, then count
    crash_by_year = df.groupby(['YEAR', 'CRASH_TYPE']).size().unstack(fill_value=0)
    
    #--------- Absolute numbers bar chart
    plt.figure(figsize=(10, 6))
    crash_by_year.plot(kind='bar', stacked=True, 
                       color=['red', 'orange', 'green'], 
                       edgecolor='black')
    plt.title('Car Crashes Yearly Distribution')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.legend(title='Crash Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    abs_fig = plt.gcf()
    
    # --------Convert counts to percentages
    crash_pct_by_year = crash_by_year.div(crash_by_year.sum(axis=1), axis=0) * 100
    
    # -----------Percentage distribution horizontal stacked bar chart
    plt.figure(figsize=(10, 6))
    crash_pct_by_year.plot(kind='barh', stacked=True, 
                           color=['red', 'orange', 'green'], 
                           edgecolor='black')
    plt.title('Crash Type Distribution by Year (%)')
    plt.xlabel('Percentage of Crashes')
    plt.ylabel('Year')
    plt.legend(title='Crash Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    pct_fig = plt.gcf()
    
    return abs_fig, pct_fig




def visualize_hourly_crash_distribution(df):
    """
    Create a line plot showing crash distribution by hour for weekdays and weekends.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'IS_WEEKEND' (0/1) and 'HOUR' columns
    
    Returns:
    --------
    matplotlib.figure.Figure
        A matplotlib figure object showing the hourly crash distribution
    
    Notes:
    ------
    - Requires pandas and matplotlib to be imported
    - Assumes 'IS_WEEKEND' column contains 0 (weekday) and 1 (weekend)
    - Assumes 'HOUR' column represents hours of the day (0-23)
    """
        
    # --------Group crashes by weekend status and hour
    hourly_dist = df.groupby(['IS_WEEKEND', 'HOUR']).size().unstack(fill_value=0)
    
    # ----------- Rename index for clarity (0 as Weekday, 1 as Weekend)
    hourly_dist.index = ['Weekday', 'Weekend']
    
    # --------- Create the plot
    plt.figure(figsize=(10, 5))
    hourly_dist.T.plot(kind='line', marker='o', figsize=(10, 5))
    
    plt.title('Crash Distribution by Hour: Weekdays vs Weekends')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Crashes')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(range(0, 24))
    plt.legend(title='Day Type')
    plt.tight_layout()
    
    
    return plt.gcf()

def visualize_monthly_crash_trend(df, datetime_column='CRASH DATETIME'):
    """
    Create a time series plot of monthly crash counts.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing a datetime column
    datetime_column : str, optional (default='CRASH DATETIME')
        Name of the column containing crash datetime information
    
    Returns:
    --------
    matplotlib.figure.Figure
        A matplotlib figure object showing monthly crash counts over time
    
    Notes:
    ------
    - Requires pandas and matplotlib to be imported
    - Assumes the datetime column can be converted to a Period by month
    """
        
    # ---------------Create year-month period column
    df['YEAR_MONTH'] = df[datetime_column].dt.to_period('M')
    
    # --------------Group by year-month and count crashes
    monthly_crashes = df.groupby('YEAR_MONTH').size()
    
    # -----------Create the plot
    plt.figure(figsize=(14, 5))
    monthly_crashes.plot(marker='o')
    
    plt.title('Monthly Crash Counts Over Time')
    plt.xlabel('Year-Month')
    plt.ylabel('Number of Crashes')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    
    return plt.gcf()


def prepare_and_plot_zipcode_crashes(df, geo_url=None):
    """
    Prepare ZIP code crash data and create a geospatial plot.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing crash data with 'ZIP CODE' column
    geo_url : str, optional
        URL for ZIP code GeoJSON. 
        Defaults to NY zip codes GeoJSON from OpenDataDE repository.
    
    Returns:
    --------
    tuple
        (matplotlib.figure.Figure, geopandas.GeoDataFrame)
        - Matplotlib figure of ZIP code crash counts
        - Prepared GeoDataFrame with crash counts
    
    Notes:
    ------
    - Requires geopandas, contextily, matplotlib, and pandas to be imported
    """
   
    # Default GeoJSON URL if not provided
    if geo_url is None:
        geo_url = "https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON/master/ny_new_york_zip_codes_geo.min.json"
    
    # Prepare ZIP code counts
    zip_counts = df['ZIP CODE'].value_counts().reset_index()
    zip_counts.columns = ['ZIP_Code', 'CRASH_COUNT']
    zip_counts['ZIP_Code'] = pd.to_numeric(zip_counts['ZIP_Code'], errors='coerce')
    
    # Load GeoJSON
    zip_shapes = gpd.read_file(geo_url)
    
    # Rename and convert ZIP code column
    zip_shapes = zip_shapes.rename(columns={'ZCTA5CE10': 'ZIP_Code'})
    zip_shapes['ZIP_Code'] = pd.to_numeric(zip_shapes['ZIP_Code'], errors='coerce')
    
    # Filter zip_shapes to only include ZIPs present in crash data
    nyc_zips = zip_counts['ZIP_Code'].dropna().unique()
    zip_shapes_nyc = zip_shapes[zip_shapes['ZIP_Code'].isin(nyc_zips)].copy()
    
    # Merge crash data
    zip_map = zip_shapes_nyc.merge(zip_counts, on='ZIP_Code', how='left')
    zip_map['CRASH_COUNT'] = zip_map['CRASH_COUNT'].fillna(0)
    
    # Convert to Web Mercator projection
    zip_map = zip_map.to_crs(epsg=3857)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    zip_map.plot(
        column='CRASH_COUNT', 
        cmap='OrRd', 
        linewidth=0.5, 
        edgecolor='gray', 
        legend=True, 
        ax=ax
    )
    
    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    
    # Styling
    ax.set_title('NYC Traffic Accidents by ZIP Code')
    ax.axis('off')
    plt.tight_layout()
    
    return fig, zip_map



# preprocess for modeling data set
def prepare_tree_dataset(df, target_column='SEVERITY_CLASS', 
                       drop_columns=None, 
                       test_size=0.2, 
                       random_state=42):
    """
    Prepare machine learning dataset with train-test split and one-hot encoding.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing features and target variable
    target_column : str, optional (default='SEVERITY_LEVEL')
        Name of the column to be used as target variable
    drop_columns : list, optional
        List of columns to drop before encoding and splitting
        If None, defaults to ['SEVERITY_CLASS', 'CRASH_TYPE', ]
    test_size : float, optional (default=0.2)
        Proportion of the dataset to include in the test split
    random_state : int, optional (default=42)
        Controls the shuffling applied to the data before splitting
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
        - Encoded and split feature training data
        - Encoded and split feature test data
        - Training target variable
        - Test target variable
    
    Notes:
    ------
    - Performs one-hot encoding on categorical variables
    - Removes specified columns before encoding
    """
        
    #       Default drop columns if not specified
    if drop_columns is None:
        drop_columns = ['SEVERITY_CLASS', 'CRASH_TYPE']
    
    # Prepare features (X) by dropping specified columns
    X = df.drop(columns=drop_columns)
    
    # Prepare target variable
    y = df[target_column]
    
    # One-hot encode features
    X_encoded = pd.get_dummies(X)
    
    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

#---------------------------------------------------------------------
#XGBoost model
def xgboost_severity_classifier(df, 
                                       target_column='SEVERITY_CLASS', 
                                       drop_columns=None, 
                                       test_size=0.2, 
                                       random_state=42, 
                                       num_classes=None,
                                       datetime_columns=None):
    """
    Prepare dataset and train an XGBoost classifier for severity level prediction.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing features and target variable
    target_column : str, optional (default='SEVERITY_CLASS')
        Name of the column to be used as target variable
    drop_columns : list, optional
        List of columns to drop before encoding and splitting
    test_size : float, optional (default=0.2)
        Proportion of the dataset to include in the test split
    random_state : int, optional (default=42)
        Random seed for reproducibility
    num_classes : int, optional
        Number of unique severity levels. If None, will be inferred from data
    datetime_columns : list, optional
        List of datetime columns to extract features from
    
    Returns:
    --------
    dict
        A dictionary containing:
        - 'model': Trained XGBoost classifier
        - 'predictions': Predicted severity levels
        - 'accuracy': Model accuracy
        - 'classification_report': Detailed classification metrics
        - 'confusion_matrix': Confusion matrix of predictions
        - 'feature_importance': Feature importance series
        - 'X_train', 'X_test', 'y_train', 'y_test': Split datasets
    
    Notes:
    ------
    - Prepares dataset using prepare_tree_dataset function
    - Adjusts labels to start from 0 for XGBoost
    - Uses multi-class classification
    - Prints and returns evaluation metrics
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_processed = df.copy()
    
    # Default drop columns if not specified
    if drop_columns is None:
        drop_columns = ['SEVERITY_CLASS', 'CRASH_TYPE']
    
    
    
    # Prepare features (X) by dropping specified columns
    X = df_processed.drop(columns=drop_columns)
    
    # Prepare target variable
    y = df_processed[target_column]
    
    # One-hot encode features
    X_encoded = pd.get_dummies(X)
    
    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Determine number of classes if not specified
    if num_classes is None:
        num_classes = len(y_train.unique())
    
    # Adjust labels to start from 0
    y_train_adj = y_train - y_train.min()
    y_test_adj = y_test - y_test.min()
    
    # Define and train the XGBoost model
    xgb_model = XGBClassifier(
        objective='multi:softmax',  # or 'multi:softprob' for probabilities
        num_class=num_classes,
        eval_metric='mlogloss',
        random_state=random_state,
        enable_categorical=True  # Handle categorical features
    )
    
    # Fit the model
    xgb_model.fit(X_train, y_train_adj)
    
    # Make predictions
    y_pred = xgb_model.predict(X_test)
    
    # Shift predictions back to original labels
    y_pred_final = y_pred + y_train.min()
    
    # Calculate accuracy
    accuracy = round(accuracy_score(y_test, y_pred_final), 4)
    
    # Generate classification report
    class_report = classification_report(y_test, y_pred_final)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_final)
    
    # Calculate feature importance
    feature_importance = pd.Series(
        xgb_model.feature_importances_, 
        index=X_train.columns
    )
    
    # Print results
    print("Accuracy: ", accuracy)
    print("\nClassification Report:\n", class_report)
    print("\nConfusion Matrix:\n", conf_matrix)
    
    # Return results for further analysis
    return {
        'model': xgb_model,
        'predictions': y_pred_final,
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'feature_importance': feature_importance,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

#-------------------------------------------------
#efatures plotting
def plot_features(feature_importance, model='Model', top_n=10):
    """
    Plot feature importances with Seaborn.
    
    Parameters:
    -----------
    feature_importance : pandas.Series
        Series of feature importances
    model : str, optional
        Name of the model for the plot title
    top_n : int, optional (default=10)
        Number of top features to plot
    
    Returns:
    --------
    matplotlib.pyplot
        Displayed plot
    """
    # Sort features and select top N
    top_features = feature_importance.sort_values(ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_features.values, y=top_features.index, palette='viridis', order=top_features.index)
    plt.title(f'Top {top_n} Feature Importances for {model}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()

    
    # Save figure
    name = f'feature_importance_{model}.png'
    plt.savefig(name, dpi=300, bbox_inches='tight')
    
    return plt.show()

#--------------------------------------------
#confusion matrix plot
def plot_confusion_matrix(conf_matrix, name):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    conf_matrix : numpy array
        Confusion matrix from the model results
    """
    plt.figure(figsize=(8,6))
    sns.heatmap(
        conf_matrix, 
        annot=True,  # Show numeric values
        fmt='d',     # Integer formatting
        cmap='Blues',  # Color scheme
        xticklabels=['No Injury', 'Injury'],
        yticklabels=['No Injury', 'Injury']
    )
    
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()
    
#------------------------------------------------------
def xgboost_severity_classifier_tr(df, 
                                target_column='SEVERITY_CLASS', 
                                drop_columns=None, 
                                test_size=0.2, 
                                random_state=42, 
                                num_classes=2,
                                # Tuning parameters
                                thresholds=None):
    """
    Train XGBoost model with threshold evaluation
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    target_column : str, optional
        Target variable column name
    drop_columns : list, optional
        Columns to drop before modeling
    test_size : float, optional
        Proportion of test set
    random_state : int, optional
        Random state for reproducibility
    num_classes : int, optional
        Number of classes
    thresholds : list, optional
        Thresholds to evaluate
    
    Returns:
    --------
    dict
        Model results and evaluation
    """
    # Prepare dataset using the prepare_tree_dataset function
    X_train, X_test, y_train, y_test = prepare_tree_dataset(
        df, 
        target_column=target_column, 
        drop_columns=drop_columns,
        test_size=test_size, 
        random_state=random_state
    )
    
    # Default thresholds if not provided
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    # Train XGBoost model
    xgb_softprob = XGBClassifier(
        objective='multi:softprob',  # probabilities for each class
        num_class=num_classes,       # number of classes
        eval_metric='mlogloss',
        random_state=random_state,
        enable_categorical=True
    )
    xgb_softprob.fit(X_train, y_train)
    
    # Predict probabilities
    p1 = xgb_softprob.predict_proba(X_test)[:, 1]
    
    # Store results
    results = {
        'model': xgb_softprob,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'thresholds': {}
    }
    
    # Evaluate at different thresholds
    for threshold in thresholds:
        # Predict based on threshold
        y_pred = (p1 >= threshold).astype(int)
        
        # Compute metrics
        results['thresholds'][threshold] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, digits=3),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # Print results
        print(f"\nThreshold = {threshold}")
        print("Accuracy:", results['thresholds'][threshold]['accuracy'])
        print("Classification Report:")
        print(results['thresholds'][threshold]['classification_report'])
        print("Confusion Matrix:")
        print(results['thresholds'][threshold]['confusion_matrix'])
    
    return results


#-------------------------------------
# xgboost with binary and tresholds:
def xgboost_class_weight_classifier( df, 
                                    target_column='SEVERITY_CLASS', 
                                    drop_columns=None, 
                                    test_size=0.2, 
                                    random_state=42, 
                                    thresholds=None):
    """
    Train XGBoost classifier with class weights and threshold analysis
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    target_column : str, optional
        Target variable column name
    drop_columns : list, optional
        Columns to drop before modeling
    test_size : float, optional
        Proportion of test set
    random_state : int, optional
        Random state for reproducibility
    thresholds : list, optional
        List of thresholds to evaluate
    
    Returns:
    --------
    dict
        Comprehensive model evaluation results
    """
    # Prepare dataset using prepare_tree_dataset function
    X_train, X_test, y_train, y_test = prepare_tree_dataset(
        df, 
        target_column=target_column, 
        drop_columns=drop_columns,
        test_size=test_size, 
        random_state=random_state
    )
    
    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(y_train),
        y=y_train
    )
    
    # Convert to dictionary
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    # Print class distribution and weights
    print("Training Class Distribution:")
    print(y_train.value_counts(normalize=True))
    print("\nClass Weights:")
    print(class_weight_dict)
    
    # Calculate scale_pos_weight
    scale_pos_weight = class_weights[1] / class_weights[0]
    
    # XGBoost with class weights
    xgb_weights = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=random_state
    )   
    
    # Train the model
    xgb_weights.fit(X_train, y_train)
    
    # Initial predictions
    y_weights = xgb_weights.predict(X_test)

    # feature importance
    # Calculate feature importance
    feature_importance = pd.Series(
    xgb_weights.feature_importances_, 
    index=X_train.columns
)
    
    # Results dictionary
    results = {
        'model': xgb_weights,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'class_weight_dict': class_weight_dict,
        'scale_pos_weight': scale_pos_weight,
        'initial_results': {
            'accuracy': accuracy_score(y_test, y_weights),
            'classification_report': classification_report(y_test, y_weights)
        },
        'feature_importance':feature_importance
    }
    
    # Print initial results
    print(f'Initial Accuracy: {results["initial_results"]["accuracy"]}')
    print("\nClassification Report before threshold adjustment:")
    print(results["initial_results"]["classification_report"])
    
    # Default thresholds if not provided
    if thresholds is None:
        thresholds = [0.5, 0.52, 0.55, 0.6]
    
    # Threshold analysis
    results['threshold_results'] = {}
    
    # Predict probabilities
    y_proba = xgb_weights.predict_proba(X_test)[:, 1]
    
    # Evaluate different thresholds
    for threshold in thresholds:
        # Predictions at current threshold
        y_pred = (y_proba >= threshold).astype(int)
        
        # Store results for this threshold
        results['threshold_results'][threshold] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # Print results
        print(f"\nThreshold: {threshold}")
        print("Accuracy:", results['threshold_results'][threshold]['accuracy'])
        print("Classification Report:")
        print(results['threshold_results'][threshold]['classification_report'])
    
    return results

# confusion matrix with threshold:
def get_confusion_matrix_at_threshold(results, threshold=0.55):
    """
    Get confusion matrix at a specific threshold
    
    Parameters:
    -----------
    results : dict
        Results from the model function
    threshold : float, optional
        Threshold for classification (default 0.55)
    
    Returns:
    --------
    numpy.ndarray
        Confusion matrix
    """
    # Predict probabilities
    y_pred_proba = results['model'].predict_proba(results['X_test'])[:, 1]
    
    # Predict based on threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(results['y_test'], y_pred)
    
    return conf_matrix



#rANDOM fOREST
def randomforest_severity_classifier(df, 
                                           target_column='SEVERITY_CLASS', 
                                           drop_columns=None, 
                                           test_size=0.2, 
                                           random_state=42, 
                                           num_classes=None,
                                           datetime_columns=None,
                                           n_estimators=100,
                                           max_depth=None):
    """
    Prepare dataset and train a Random Forest classifier for severity level prediction.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing features and target variable
    target_column : str, optional (default='SEVERITY_CLASS')
        Name of the column to be used as target variable
    drop_columns : list, optional
        List of columns to drop before encoding and splitting
    test_size : float, optional (default=0.2)
        Proportion of the dataset to include in the test split
    random_state : int, optional (default=42)
        Random seed for reproducibility
    num_classes : int, optional
        Number of unique severity levels. If None, will be inferred from data
    datetime_columns : list, optional
        List of datetime columns to extract features from
    n_estimators : int, optional (default=100)
        Number of trees in the forest
    max_depth : int, optional (default=None)
        Maximum depth of the trees
    
    Returns:
    --------
    dict
        A dictionary containing:
        - 'model': Trained Random Forest classifier
        - 'predictions': Predicted severity levels
        - 'accuracy': Model accuracy
        - 'classification_report': Detailed classification metrics
        - 'confusion_matrix': Confusion matrix of predictions
        - 'feature_importance': Feature importance series
        - 'X_train', 'X_test', 'y_train', 'y_test': Split datasets
    
    Notes:
    ------
    - Prepares dataset with one-hot encoding
    - Adjusts labels to start from 0 if needed
    - Uses multi-class classification
    - Prints and returns evaluation metrics
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_processed = df.copy()
    
    # Default drop columns if not specified
    if drop_columns is None:
        drop_columns = ['SEVERITY_CLASS', 'CRASH_TYPE']
    
    
    
    # Prepare features (X) by dropping specified columns
    X = df_processed.drop(columns=drop_columns)
    
    # Prepare target variable
    y = df_processed[target_column]
    
    # One-hot encode features
    X_encoded = pd.get_dummies(X)
    
    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Determine number of classes if not specified
    if num_classes is None:
        num_classes = len(y_train.unique())
    
    # Adjust labels to start from 0 if needed
    label_offset = y_train.min()
    y_train_adj = y_train - label_offset
    y_test_adj = y_test - label_offset
    
    # Define and train the Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )
    
    # Fit the model
    rf_model.fit(X_train, y_train_adj)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Shift predictions back to original labels
    y_pred_final = y_pred + label_offset
    
    # Calculate accuracy
    accuracy = round(accuracy_score(y_test, y_pred_final), 4)
    
    # Generate classification report
    class_report = classification_report(y_test, y_pred_final)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_final)
    
    # Calculate feature importance
    feature_importance = pd.Series(
        rf_model.feature_importances_, 
        index=X_train.columns
    )
    
    # Print results
    print("Accuracy: ", accuracy)
    print("\nClassification Report:\n", class_report)
    print("\nConfusion Matrix:\n", conf_matrix)
    
    # Return results for further analysis
    return {
        'model': rf_model,
        'predictions': y_pred_final,
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'feature_importance': feature_importance,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


