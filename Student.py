import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

np.random.seed(52)


class RandomForestClassifier:
    def __init__(self, n_trees=10, max_depth=np.iinfo(np.int64).max, min_error=1e-6, random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_error = min_error
        self.random_state = random_state
        self.forest = []
        self.is_fit = False

    # Step 2
    def create_bootstrap(self, X, y):
        mask = np.random.choice(len(X), size=len(X), replace=True)
        return X[mask], y[mask]

    def fit(self, X, y):

        for _ in tqdm(range(self.n_trees), desc='Training Trees'):
            X_sample, y_sample = self.create_bootstrap(X, y)
            tree = DecisionTreeClassifier(max_depth=self.max_depth, max_features='sqrt',
                                          min_impurity_decrease=self.min_error, random_state=self.random_state)
            tree.fit(X_sample, y_sample)
            self.forest.append(tree)

            self.is_fit = True

    def predict(self, testing_set):
        if not self.is_fit:
            raise AttributeError("The forest is not fit yet! Consider calling .fit() method.")


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

    # Step 2 implement
    # X_bs, y_bs = create_bootstrap(X_train, y_train)
    # print(list(y_bs[0:10]))

    # Stage 1
    # clf = DecisionTreeClassifier()
    # clf.fit(X_train, y_train)
    # prediction_X_val = clf.predict(X_val)
    # test_score = accuracy_score(y_val, prediction_X_val)
    # print(round(test_score, 3))

    rfc = RandomForestClassifier(n_trees=10, max_depth=4, min_error=1e-6)
    rfc.fit(X_train, y_train)
    predictions = rfc.forest[0].predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    print(round(accuracy, 3))



