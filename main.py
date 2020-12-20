from sample.dataset import load_transformed_car_data, plot_feature_importance
from sample.machine_learning import split_test_and_train, linear_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def main():
    car_data = load_transformed_car_data()

    y = car_data.pop('Selling_Price')
    X = car_data

    plot_feature_importance(X, y)

    model_sets = split_test_and_train(X, y)

    reg = LinearRegression()

    linear_fit(reg, X, y, model_sets, 3)
    linear_fit(reg, X, y, model_sets, 4)
    linear_fit(reg, X, y, model_sets, 5)
    linear_fit(reg, X, y, model_sets, 6)

    print('k=5 has the smallest mean squared error')


if __name__ == "__main__":
    main()
