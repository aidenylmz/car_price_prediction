from sample.dataset import load_transformed_car_data, plot_feature_importance
from sample.machine_learning import split_test_and_train, linear_fit


def main():
    car_data = load_transformed_car_data()

    y = car_data.pop('Selling_Price')
    X = car_data

    plot_feature_importance(X, y)

    model_sets = split_test_and_train(X, y)

    linear_fit(model_sets)


if __name__ == "__main__":
    main()
