from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt


def split_test_and_train(x, y):
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)

    return {"X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test}


def linear_fit(model, x, y, model_sets, k=5):
    model.fit(model_sets['X_train'], model_sets['y_train'])
    y_pred = model.predict(model_sets['X_test'])

    rsqrd = r2_score(model_sets['y_test'], y_pred)
    print(f'k = {k} (Cross-validation splitting)')
    print(f'R2 Error: {rsqrd}')

    scores = cross_val_score(
        model, x, y, scoring='neg_mean_squared_error', cv=k)
    rmse = np.sqrt(-scores)
    print('Reg rmse:', rmse)
    print('Reg Mean:', rmse.mean())
    print('---------------------------------------')

    plt.figure(figsize=(18, 8))
    sns.histplot(model_sets['y_test'] - y_pred)
    plt.savefig('docs/distplot.png')

    plt.figure(figsize=(10, 5))
    plt.scatter(model_sets['y_test'], y_pred)
    plt.savefig('docs/scatter.png')
