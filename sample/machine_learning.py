from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def split_test_and_train(x, y):
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)

    return {"X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test}


def linear_fit(model_sets):
    reg = LinearRegression()
    reg.fit(model_sets['X_train'], model_sets['y_train'])
    y_pred = reg.predict(model_sets['X_test'])

    rsqrd = r2_score(model_sets['y_test'], y_pred)
    print(f'R2 Error: {rsqrd}')
