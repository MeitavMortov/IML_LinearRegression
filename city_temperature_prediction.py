
import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
CITY_TEMP_FILENAME = '../datasets/city_temperature.csv'
COUNTRIES = ['South Africa', 'The Netherlands', 'Jordan']
DEGREES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
BEST_DEGREE = 5


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    # Load city daily temperature dataset, Set the type of the 'Date' column:
    data_frame = pd.read_csv(filename, parse_dates=['Date'])
    # Remove missing values:
    data_frame = data_frame.dropna()
    # Remove duplicate rows:
    data_frame = data_frame.drop_duplicates()
    # Deal with invalid/ irrelevant data:
    data_frame = data_frame[0 < data_frame.Temp]
    # Add a `DayOfYear` column based on the `Date` column:
    date_column = data_frame['Date']
    # Find orginal day of the year://pandas.pydata.org/docs/reference/api/pandas.Series.dt.dayofyear.html
    data_frame['DayOfYear'] = date_column.dt.dayofyear
    return data_frame
def explore_israel(data_frame):
    """
    Subset the dataset to contain samples only from the country of Israel. Investigate how the
    average daily temperature change as a function of the `DayOfYear`-
    Implementation  for Question 2.
    :return: nothing.
    """
    israel_data = data_frame[data_frame.Country == 'Israel']
    # Color code the dots by the different years:
    color_year = israel_data["Year"].astype(str)  # Color scale is discrete if year is string and not integer.
    title_1 = 'The relation between average daily temperature to DayOfYear in Israel:'
    # Plot a scatter plot showing the relation between average daily temperature to DayOfYear:
    fig_1 = px.scatter(data_frame=israel_data, x='DayOfYear', y='Temp', color=color_year, title=title_1)
    fig_1.write_image("Q2.1PartB.png")
    # Group the samples by `Month`:
    group_by_month = israel_data.groupby(by=['Month'], axis=0, level=None, as_index=False, sort=True,
                                         group_keys=True, observed=False, dropna=True)
    # https://datascienceparichay.com/article/pandas-groupby-standard-deviation/
    aggregated_israel_data = group_by_month.agg(std=("Temp", "std"))
    # Plot a bar plot showing for each month the standard deviation of the daily temperatures:
    title_2 = 'The standard deviation of the daily temperatures:'
    fig_2 = px.bar(data_frame=aggregated_israel_data, x='Month', y='std',
                   text=np.round(aggregated_israel_data['std'],2))
    fig_2.update_layout(title=title_2, yaxis_title='Standard Deviation')
    fig_2.write_image("Q2.2PartB.png")
    return

def explore_countries(data_frame):
    """
    Group the samples according to `Country` and `Month` and calculate the average and standard deviation
    of the temperature. Plot a line plot of the average monthly temperature, with error
     bars color coded by the country.
    Implementation  for Question 3.
    :return: nothing.
    """
    # Group the samples by `Country` and `Month`:
    group_by_country_and_month = data_frame.groupby(by=['Country', 'Month'], axis=0, level=None,
                                                    as_index=False, sort=True,
                                         group_keys=True, observed=False, dropna=True)
    # Calculate the average:
    aggregated_data = group_by_country_and_month.agg(mean=("Temp", "mean"), std=("Temp", "std"))
    title = 'The average temperature of each month'
    #Color coded by the country:
    color_country = 'Country'
    # Plot a line plot of the average monthly temperature:
    fig = px.line(data_frame=aggregated_data, x='Month', y='mean', color=color_country, error_y='std')
    fig.update_layout(title=title, yaxis_title='Standard Deviation')
    fig.write_image("Q3PartB.png")

def fitting_model_for_israel(data_frame):
    """
    Group the samples according to `Country` and `Month` and calculate the average and standard deviation
    of the temperature. Plot a line plot of the average monthly temperature, with error bars color coded
    by the country.
    Implementation  for Question 4.
    :return: nothing.
    """
    # Over the subset containing observations only from Israel:
    israel_data = data_frame[data_frame.Country == 'Israel']
    # Randomly split the dataset into a training set (75%) and test set (25%):
    train_X, train_y, test_X, test_y = \
        split_train_test(X=israel_data.DayOfYear, y=israel_data.Temp, train_proportion=0.75)
    # Convert the DataFrame to a NumPy array:
    train_X_arr, train_y_arr, test_X_arr, test_y_arr = train_X.to_numpy(),\
        train_y.to_numpy(), test_X.to_numpy(), test_y.to_numpy()
    # For every value k ∈ [1,10], fit a polynomial model of degree k using the training set:
    loss_dict = {}
    for deg in DEGREES:
        polynomial_model = PolynomialFitting(deg)
        #Fit the polynomial model:
        polynomial_model.fit(train_X_arr, train_y_arr)
        # Record the loss of the model over the test set, rounded to 2 decimal places:
        loss_of_curr_deg = polynomial_model.loss(test_X_arr, test_y_arr)
        loss_dict[deg] = np.round(loss_of_curr_deg, 2)
    # Print the test error recorded for each value of k:
    print(loss_dict)
    # plot a bar plot showing the test error recorded for each value of k
    title = 'The test error recorded for different degrees of polynomial model: '
    curr_df = pd.DataFrame({'Keys': list(loss_dict.keys()),
        'Values': list(loss_dict.values())})
    fig = px.bar(data_frame=curr_df, x=list(loss_dict.keys()), y=list(loss_dict.values()),
                 color=curr_df['Values'].astype(str),  text='Values')
    fig.update_layout(title=title, xaxis_title='Degree of polynomial model', yaxis_title='Loss')
    fig.write_image("Q4PartB.png")

def evaluate_fitted_model_diff_countries(data_frame):
    """
    Fit a model over the entire subset of records from Israel using the k chosen above. Plot a bar
    plot showing the model’s error over each of the other countries
    Implementation  for Question 5:
    """
    # Model records from Israel with k=5 because then loss is minimal.
    israel_data = data_frame[data_frame.Country == 'Israel']
    X = israel_data.DayOfYear.to_numpy()
    y = israel_data.Temp.to_numpy()
    polynomial_model = PolynomialFitting(BEST_DEGREE)
    # Fit the polynomial model:
    polynomial_model.fit(X=X, y=y)
    loss_dict = {}
    for country in COUNTRIES:
        country_data = data_frame[data_frame.Country == country]
        loss = polynomial_model.loss(country_data.DayOfYear, country_data.Temp)
        loss_dict[country] = np.round(loss, 2)
    title = 'The fitted over Israel model’s error over each of the countries: '
    curr_df = pd.DataFrame({'Keys': list(loss_dict.keys()),
        'Values': list(loss_dict.values())})
    fig = px.bar(data_frame=curr_df, x=list(loss_dict.keys()), y=list(loss_dict.values()),
                 color=curr_df['Values'].astype(str), text='Values')
    fig.update_layout(title=title, xaxis_title='Country', yaxis_title='Loss')
    fig.write_image("Q5PartB.png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data_frame = load_data(CITY_TEMP_FILENAME)

    # Question 2 - Exploring data for specific country
    explore_israel(data_frame)

    # Question 3 - Exploring differences between countries
    explore_countries(data_frame)

    # Question 4 - Fitting model for different values of `k`
    fitting_model_for_israel(data_frame)

    # Question 5 - Evaluating fitted model on different countries
    evaluate_fitted_model_diff_countries(data_frame)