import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from scipy.stats import mode

np.random.seed(52)


class RandomForestClassifier:
    def __init__(self, n_trees=1, max_depth=np.iinfo(np.int64).max, min_error=1e-6):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_error = min_error
        self.forest = []
        self.is_fit = False

    # Step 2
    def create_bootstrap(self, training_set, testing_set):
        mask = np.random.choice(len(training_set), size=len(training_set), replace=True)
        return training_set[mask], testing_set[mask]

    def fit(self, training_set, testing_set):

        for _ in tqdm(range(self.n_trees), desc="Training Trees"):
            X_sample, y_sample = self.create_bootstrap(training_set, testing_set)
            tree = DecisionTreeClassifier(max_depth=self.max_depth, max_features='sqrt',
                                          min_impurity_decrease=self.min_error)
            tree.fit(X_sample, y_sample)
            self.forest.append(tree)

            self.is_fit = True

    def predict(self, testing_set):
        if not self.is_fit:
            raise AttributeError("The forest is not fit yet! Consider calling .fit() method.")
        prediction = np.array([tree.predict(testing_set) for tree in self.forest])
        majority_votes, _ = mode(prediction, axis=0)
        return majority_votes.ravel()


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

    # Stage 2 implement
    # X_bs, y_bs = create_bootstrap(X_train, y_train)
    # Stage 2 output
    # print(list(y_bs[0:10]))

    # Stage 1
    # clf = DecisionTreeClassifier()
    # clf.fit(X_train, y_train)
    # prediction_X_val = clf.predict(X_val)
    # test_score = accuracy_score(y_val, prediction_X_val)
    # Stage 1 output

    # rfc = RandomForestClassifier(n_trees=30, max_depth=11, min_error=1e-6)
    # rfc.fit(X_train, y_train)
    # predictions = rfc.predict(X_val)
    # accuracy = accuracy_score(y_val, predictions)

    # Stage 3 output
    # print(round(accuracy, 3))

    # Stage 4 output
    # print(list(predictions[0:10]))

    # Stage 5 (1st stage output: 0.797)
    # print('RandomForestClassifier accuracy:', round(accuracy, 3))
    # print('Single DecisionTree accuracy:', round(test_score, 3))

    # Stage 6
    number_of_trees = 1
    resulting_accuracy = []
    number_of_trees_list = []

    while number_of_trees < 600:
        rfc = RandomForestClassifier(n_trees=number_of_trees)
        rfc.fit(X_train, y_train)
        predictions = rfc.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        rounded_accuracy = round(accuracy, 3)
        resulting_accuracy.append(rounded_accuracy)
        number_of_trees_list.append(number_of_trees)
        number_of_trees += 1

    print(resulting_accuracy[0:20])

    fig, ax = plt.subplots(figsize=(100, 40))
    fig.suptitle("Dependence of accuracy from the number of trees", fontsize=70)
    ax.plot(number_of_trees_list, resulting_accuracy, c='r', linewidth=4)
    ax.set_xlabel("Number of trees", fontsize=50)
    ax.set_ylabel("Accuracy", fontsize=50)
    ax.tick_params(axis='both', which='major', labelsize=35)
    plt.show()
