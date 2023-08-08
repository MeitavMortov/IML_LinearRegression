
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
from typing import Optional
DATA_PATH = "../datasets/house_prices.csv"


from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

TRAIN_PROPORTION = 0.75  # Q1

MEAN_DICT = {} #Q2
IRRELEVANT_FEATURES = ["id", "lat", "long", "date", "sqft_lot15",'yr_built','yr_renovated', "sqft_living15"] # Q2
MAX_BEDROOMS = 18  # Q2
MAX_SQFT_LOT = 1651359 # Q2
FEATURES_LIST = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
                 "waterfront", "view", "condition","grade",
                 "sqft_above", "sqft_basement"]  # Q3

FIRST_P = 10 # Q4
NUM_OF_P = 91 # Q4
LAST_P = 100 # Q4
NUM_OF_TIMES = 10 # Q4
COLOR = 'rgb(31, 119, 180)'  # Q4
SIDE_COLOR = "#444"  # Q4
FILL_COLOR = 'rgba(68, 68, 68, 0.3)'  # Q4

def init_mean_dict(df: pd.DataFrame):
    """
    Helper function to preprocess_data that calculate in each colmoun with its mean.
    Parameters
    ----------
    X  DataFrame
    return data frame with same shape in which nan values in each colmoun with the mean.
    """
    l = FEATURES_LIST + ['zipcode']
    for feature in l:
        mean_value = df[feature].mean()
        MEAN_DICT[feature] = mean_value

def replace_nan_with_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to preprocess_data that removes nan values in each colmoun with the mean.
    Parameters
    ----------
    X  DataFrame
    return data frame with same shape in which nan values in each colmoun with the mean.
    """
    # https://www.geeksforgeeks.org/how-to-fill-nan-values-with-mean-in-pandas/
    l = FEATURES_LIST + ['zipcode']
    for feature in l:
        df[feature].fillna(value= MEAN_DICT[feature], inplace=True)
    return df

def remove_irrelevant_features(df) -> pd.DataFrame:
    df = df.drop(IRRELEVANT_FEATURES, axis=1)
    return df

def deal_with_zipcode(X) -> pd.DataFrame:
    X['zipcode'] = X['zipcode'].astype(int)
    X = pd.get_dummies(X, prefix='zipcode_', columns=['zipcode'])
    return  X

def deal_with_invalid_vals(X, is_train) -> pd.DataFrame:
    """
     Treat invalid values in the train and the test set.
     I decided validity according to  https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
    Parameters
    ----------
    X- the data frame
    is_train- boolean. true if X is train set, false if is the test set
    Returns X updated to contain only valid values. if train set remove invalid. if test set replace with mean.
    -------
    """
    if is_train:
        X = X[X['price'] > 0]
        X = X[X['sqft_living'] > 0]
        X = X[X['sqft_lot'] > 0]
        X = X[X['sqft_above'] > 0]
        X = X[X['sqft_basement'] >= 0]
        X = X[X['floors'] >= 0]
        X = X[X['bathrooms'] >= 0]
        X = X[(X['view'] >= 0) & (X['view'] <= 4)]
        X = X[(X['grade'] >= 1) & (X['grade'] <= 14)]
        X = X[(X['condition'] >= 1) & (X['condition'] <= 5)]
        X = X[(X['waterfront'] == 0) | (X['waterfront'] == 1)]
        X = X[X['bedrooms'] < MAX_BEDROOMS]
        X = X[X["sqft_lot"] < MAX_SQFT_LOT]
    else: #https://datatofish.com/if-condition-in-pandas-dataframe/
        X.loc[X['sqft_living'] <= 0, 'sqft_living'] = MEAN_DICT['sqft_living']
        X.loc[X['sqft_lot'] <= 0, 'sqft_lot'] = MEAN_DICT['sqft_lot']
        X.loc[X['sqft_above'] <= 0, 'sqft_above'] = MEAN_DICT['sqft_above']
        X.loc[X['sqft_basement'] < 0, 'sqft_basement'] = MEAN_DICT['sqft_basement']
        X.loc[X['floors'] < 0, 'floors'] = MEAN_DICT['floors']
        X.loc[X['bathrooms'] < 0, 'bathrooms'] = MEAN_DICT['bathrooms']
        X.loc[(X['view'] < 0) | (X['view'] > 4),'view'] = MEAN_DICT['view']
        X.loc[(X['grade'] < 1) | (X['grade'] > 14), 'grade'] = MEAN_DICT['grade']
        X.loc[(X['condition'] < 1) | (X['condition'] > 5), 'condition'] = MEAN_DICT['condition']
        X.loc[(X['waterfront'] != 0) & (X['waterfront'] != 1), 'waterfront'] = MEAN_DICT['waterfront']
        X.loc[X['bedrooms'] >= MAX_BEDROOMS, 'bedrooms'] = MEAN_DICT['bedrooms']
        X.loc[X["sqft_lot"] >= MAX_SQFT_LOT, "sqft_lot"] = MEAN_DICT['sqft_lot']
    return X

def preprocess_train_set(X: pd.DataFrame, y: pd.Series)-> (pd.DataFrame,pd.Series):
    """
    preprocess train data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    y : array-like of shape (n_samples, )
        Response vector corresponding given samples
    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]-
    """
    # first of all stack y- then if I will remove a row the matching price also will be removed:
    X['price'] = y
    # Remove lines with nan:
    X = X.dropna()
    # Remove duplicates lines
    X = X.drop_duplicates()
    # Remove Irrelevant features:
    X = remove_irrelevant_features(X)
    # for each feature check all values in the correct range - remove them
    # according to https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
    X = deal_with_invalid_vals(X, is_train=True)
    X = deal_with_zipcode(X)
    return X.drop('price', axis=1), X.price


def preprocess_test_set(X: pd.DataFrame)-> (pd.DataFrame,None):
    """
    preprocess test data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    Returns
    -------
    Post-processed design matrix and response vector None
    """
    # if nan values in one colmon replace them with the mean
    X = replace_nan_with_mean(X)
    # drop colmons with Irrelevant data:
    X = remove_irrelevant_features(X)
    # if invalid replace with mean:
    X = deal_with_invalid_vals(X, is_train=False)
    X = deal_with_zipcode(X)
    return X, None

def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None)-> (pd.DataFrame,Optional[pd.Series]):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    y : array-like of shape (n_samples, )
        Response vector corresponding given samples
    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    if y is None:
        # test set
        return preprocess_test_set(X)
    # train set:
    return preprocess_train_set(X, y)


def person_correlation(x, y):
    """
       Compute Pearson Correlation between given features x and y.
    """
    covariance = np.cov(x, y)
    standard_deviation_x_and_y = np.std(x) * np.std(y)
    return covariance[0, 1] * (1 / standard_deviation_x_and_y)


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    y : array-like of shape (n_samples, )
        Response vector to evaluate against
    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in FEATURES_LIST:
        person_correlation_res = person_correlation(X[feature], y)
        df = pd.DataFrame({f" {feature} ": X[feature], "Response": y})
        title = f"Pearson Correlation Between {feature} and the Response {round(person_correlation_res, 4)}:"
        #https://plotly.com/python/linear-fits/
        feature_figure = px.scatter(df, x=f" {feature} ", y="Response", title=title, trendline="ols", trendline_color_override="grey")
        feature_figure.write_image(output_path + f"Q3PartA_{feature}.png")
    return
def fit_model_over_increasing_p(p_train_X, p_train_y, p_test_X, p_test_y):
    """
    # Question 4 implementation- Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    :param p_train_X: Post-processed train_X
    :param p_train_y: Post-processed train_y
    :param p_test_X: Post-processed test_X
    :param p_test_y: Post-processed test_y
    :return: nothing
    """
    curr_p = FIRST_P
    loss_arr = np.empty([NUM_OF_P, NUM_OF_TIMES])
    for i in range(NUM_OF_P):
        for time in range(NUM_OF_TIMES):
            # Step 1:
            frac = curr_p / 100
            samples_X = p_train_X.sample(frac=frac)
            X_indexes = samples_X.index
            samples_y = p_train_y.loc[X_indexes]
            # Step 2:
            linear_regression_model = LinearRegression(True)
            linear_regression_model.fit(samples_X, samples_y)
            # Step 3:
            curr_loss = linear_regression_model.loss(p_test_X, p_test_y)
            loss_arr[i, time] = curr_loss
        curr_p = curr_p + 1
    # Step 4:
    mean = np.mean(a=loss_arr, axis=1)
    std = np.std(a=loss_arr, axis=1)
    percentage_arr = [i for i in range(NUM_OF_TIMES, LAST_P+1)]
    # Step 5: I used https://plotly.com/python/continuous-error-bars/
    uper_scater = go.Scatter(x=percentage_arr, y=mean + (2 * std), mode='lines', marker=dict(color=SIDE_COLOR),
        line=dict(width=0), showlegend=False)  # mean+2*std
    lower_scater = go.Scatter(x=percentage_arr, y=mean - (2 * std), marker=dict(color=SIDE_COLOR), line=dict(width=0),
        mode='lines', fillcolor=FILL_COLOR, fill='tonexty', showlegend=False)  # mean-2*std
    orig_scater = go.Scatter(x=percentage_arr, y=mean, mode='lines', line=dict(color=COLOR))  # mean
    fig = go.Figure([uper_scater, lower_scater, orig_scater])
    fig.update_layout(xaxis_title= 'Percentage of Training data', yaxis_title='Average loss',
                      title='Average loss as function of training size with error ribbon of size: ')
    fig.write_image("Q4PartA.png")


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv(DATA_PATH)
    # According to the forum we can remove nan prices:
    df = df[df['price'].notna()]
    # Question 1 - split data into train and test sets
    X = df.drop(['price'], axis=1)
    y = df.price
    train_X, train_y, test_X, test_y = split_train_test(X, y, TRAIN_PROPORTION)
    # Question 2 - Preprocessing of housing prices dataset
    init_mean_dict(train_X)
    p_train_X, p_train_y = preprocess_data(train_X, train_y)
    p_test_X = preprocess_data(test_X, None)[0]
    p_test_X = p_test_X.reindex(columns=p_train_X.columns, fill_value=0)
    p_test_y = test_y
    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(p_train_X, p_train_y, ".")
    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    fit_model_over_increasing_p(p_train_X, p_train_y, p_test_X, p_test_y)


