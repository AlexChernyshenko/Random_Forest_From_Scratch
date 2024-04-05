import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

np.random.seed(52)


def create_bootstrap(training_set, testing_set):
    mask = np.random.choice(len(training_set), size=len(training_set), replace=True)
    X_bootstrap = training_set[mask]
    y_bootstrap = testing_set[mask]
    return X_bootstrap, y_bootstrap


def convert_embarked(x):
    if x == 'S':
        return 0
    elif x == 'C':
        return 1
    else:
        return 2


if __name__ == '__main__':
    data = pd.read_csv('https://www.dropbox.com/s/4vu5j6ahk2j3ypk/titanic_train.csv?dl=1')

    data.drop(
        ['PassengerId', 'Name', 'Ticket', 'Cabin'],
        axis=1,
        inplace=True
    )
    data.dropna(inplace=True)

    # Separate these back
    y = data['Survived'].astype(int)
    X = data.drop('Survived', axis=1)

    X['Sex'] = X['Sex'].apply(lambda x: 0 if x == 'male' else 1)
    X['Embarked'] = X['Embarked'].apply(lambda x: convert_embarked(x))

    X_train, X_val, y_train, y_val = \
        train_test_split(X.values, y.values, stratify=y, train_size=0.8)

    X_bs, y_bs = create_bootstrap(X_train, y_train)
    print(list(y_bs[0:10]))

    # Stage 1
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    prediction_X_val = clf.predict(X_val)
    test_score = accuracy_score(y_val, prediction_X_val)
    # print(round(test_score, 3))
