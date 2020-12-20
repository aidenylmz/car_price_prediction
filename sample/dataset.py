import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from helpers.extensions import sort_features


def load_transformed_car_data():
    '''
    It loads the car data from docs, prints some descriptions,
    drops car name and year (after calculating age from year) columns,
    converts categorical columns into indicators, and saves heatmap file.
    '''
    data = pd.read_csv(
        'docs/car_data.csv')

    all_cols = data.columns
    numerical_cols = data._get_numeric_data().columns.to_list()
    categorical_cols = list(set(all_cols) - set(numerical_cols))

    (row_size, column_size) = data.shape
    print(f'Rows: {row_size}, Columns: {column_size}.')

    print(f'Numerical columns: {numerical_cols}\n')
    print(f'Categorical columns: {categorical_cols}\n')

    data.drop(['Car_Name'], inplace=True, axis=1)

    this_year = datetime.today().year

    data['Car_Age'] = this_year - data.Year
    data.drop(['Year'], inplace=True, axis=1)

    print('Before converting categorical column')
    print('---------------------------------------')
    print(data.head())
    print('---------------------------------------\n')

    data = pd.get_dummies(data, drop_first=True)

    print('After converting categorical column')
    print('---------------------------------------')
    print(data.head())
    print('---------------------------------------')

    plt.figure(figsize=(18, 8))
    sns.heatmap(data.corr(), annot=True)
    plt.xticks(rotation=0, fontsize=8)
    plt.yticks(fontsize=8)
    plt.savefig('docs/correlation_heatmap.png')

    return data


def plot_feature_importance(x, y):

    model = ExtraTreesRegressor()
    model.fit(x, y)
    plt.figure(figsize=(18, 8))

    sorted_features = sort_features(
        model.feature_importances_, y.columns)

    sns.barplot(x=sorted_features, y=sorted(
        model.feature_importances_, reverse=True))
    plt.savefig('docs/feature_importance.png')
